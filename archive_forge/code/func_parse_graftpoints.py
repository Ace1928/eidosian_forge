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
def parse_graftpoints(graftpoints: Iterable[bytes]) -> Dict[bytes, List[bytes]]:
    """Convert a list of graftpoints into a dict.

    Args:
      graftpoints: Iterator of graftpoint lines

    Each line is formatted as:
        <commit sha1> <parent sha1> [<parent sha1>]*

    Resulting dictionary is:
        <commit sha1>: [<parent sha1>*]

    https://git.wiki.kernel.org/index.php/GraftPoint
    """
    grafts = {}
    for line in graftpoints:
        raw_graft = line.split(None, 1)
        commit = raw_graft[0]
        if len(raw_graft) == 2:
            parents = raw_graft[1].split()
        else:
            parents = []
        for sha in [commit, *parents]:
            check_hexsha(sha, 'Invalid graftpoint')
        grafts[commit] = parents
    return grafts