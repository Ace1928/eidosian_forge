import hashlib
def calculate_multipart_etag(source_path, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    calculates a multipart upload etag for amazon s3

    Arguments:

    source_path -- The file to calculate the etag for
    chunk_size -- The chunk size to calculate for.
    """
    md5s = []
    with open(source_path, 'rb') as fp:
        while True:
            data = fp.read(chunk_size)
            if not data:
                break
            md5 = hashlib.new('md5', usedforsecurity=False)
            md5.update(data)
            md5s.append(md5)
    if len(md5s) == 1:
        new_etag = f'"{md5s[0].hexdigest()}"'
    else:
        digests = b''.join((m.digest() for m in md5s))
        new_md5 = hashlib.md5(digests)
        new_etag = f'"{new_md5.hexdigest()}-{len(md5s)}"'
    return new_etag