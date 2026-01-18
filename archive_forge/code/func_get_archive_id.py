import hashlib
from boto.glacier.utils import chunk_hashes, tree_hash, bytes_to_hex
from boto.glacier.utils import compute_hashes_from_fileobj
def get_archive_id(self):
    self.close()
    return self.uploader.archive_id