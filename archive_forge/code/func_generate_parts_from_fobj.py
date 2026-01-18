import hashlib
from boto.glacier.utils import chunk_hashes, tree_hash, bytes_to_hex
from boto.glacier.utils import compute_hashes_from_fileobj
def generate_parts_from_fobj(fobj, part_size):
    data = fobj.read(part_size)
    while data:
        yield data.encode('utf-8')
        data = fobj.read(part_size)