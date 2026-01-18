import hashlib
from boto.glacier.utils import chunk_hashes, tree_hash, bytes_to_hex
from boto.glacier.utils import compute_hashes_from_fileobj
@property
def current_tree_hash(self):
    """
        Returns the current tree hash for the data that's been written
        **so far**.

        Only once the writing is complete is the final tree hash returned.
        """
    return tree_hash(self.uploader._tree_hashes)