import hashlib
from boto.glacier.utils import chunk_hashes, tree_hash, bytes_to_hex
from boto.glacier.utils import compute_hashes_from_fileobj
@property
def current_uploaded_size(self):
    """
        Returns the current uploaded size for the data that's been written
        **so far**.

        Only once the writing is complete is the final uploaded size returned.
        """
    return self.uploader._uploaded_size