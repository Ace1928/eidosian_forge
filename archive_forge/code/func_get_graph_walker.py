import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
def get_graph_walker(self, heads: Optional[List[ObjectID]]=None) -> ObjectStoreGraphWalker:
    """Retrieve a graph walker.

        A graph walker is used by a remote repository (or proxy)
        to find out which objects are present in this repository.

        Args:
          heads: Repository heads to use (optional)
        Returns: A graph walker object
        """
    if heads is None:
        heads = [sha for sha in self.refs.as_dict(b'refs/heads').values() if sha in self.object_store]
    parents_provider = ParentsProvider(self.object_store)
    return ObjectStoreGraphWalker(heads, parents_provider.get_parents, shallow=self.get_shallow())