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
def fetch_pack_data(self, determine_wants, graph_walker, progress, get_tagged=None, depth=None):
    """Fetch the pack data required for a set of revisions.

        Args:
          determine_wants: Function that takes a dictionary with heads
            and returns the list of heads to fetch.
          graph_walker: Object that can iterate over the list of revisions
            to fetch and has an "ack" method that will be called to acknowledge
            that a revision is present.
          progress: Simple progress function that will be called with
            updated progress strings.
          get_tagged: Function that returns a dict of pointed-to sha ->
            tag sha for including tags.
          depth: Shallow fetch depth
        Returns: count and iterator over pack data
        """
    missing_objects = self.find_missing_objects(determine_wants, graph_walker, progress, get_tagged, depth=depth)
    remote_has = missing_objects.get_remote_has()
    object_ids = list(missing_objects)
    return (len(object_ids), generate_unpacked_objects(self.object_store, object_ids, progress=progress, other_haves=remote_has))