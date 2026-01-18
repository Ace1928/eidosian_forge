import hashlib
from boto.glacier.utils import chunk_hashes, tree_hash, bytes_to_hex
from boto.glacier.utils import compute_hashes_from_fileobj
def resume_file_upload(vault, upload_id, part_size, fobj, part_hash_map, chunk_size=_ONE_MEGABYTE):
    """Resume upload of a file already part-uploaded to Glacier.

    The resumption of an upload where the part-uploaded section is empty is a
    valid degenerate case that this function can handle. In this case,
    part_hash_map should be an empty dict.

    :param vault: boto.glacier.vault.Vault object.
    :param upload_id: existing Glacier upload id of upload being resumed.
    :param part_size: part size of existing upload.
    :param fobj: file object containing local data to resume. This must read
        from the start of the entire upload, not just from the point being
        resumed. Use fobj.seek(0) to achieve this if necessary.
    :param part_hash_map: {part_index: part_tree_hash, ...} of data already
        uploaded. Each supplied part_tree_hash will be verified and the part
        re-uploaded if there is a mismatch.
    :param chunk_size: chunk size of tree hash calculation. This must be
        1 MiB for Amazon.

    """
    uploader = _Uploader(vault, upload_id, part_size, chunk_size)
    for part_index, part_data in enumerate(generate_parts_from_fobj(fobj, part_size)):
        part_tree_hash = tree_hash(chunk_hashes(part_data, chunk_size))
        if part_index not in part_hash_map or part_hash_map[part_index] != part_tree_hash:
            uploader.upload_part(part_index, part_data)
        else:
            uploader.skip_part(part_index, part_tree_hash, len(part_data))
    uploader.close()
    return uploader.archive_id