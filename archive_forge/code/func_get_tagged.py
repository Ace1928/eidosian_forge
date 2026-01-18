import collections
import os
import socket
import sys
import time
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple
import socketserver
import zlib
from dulwich import log_utils
from .archive import tar_stream
from .errors import (
from .object_store import peel_sha
from .objects import Commit, ObjectID, valid_hexsha
from .pack import ObjectContainer, PackedObjectContainer, write_pack_from_container
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, RefsContainer, write_info_refs
from .repo import BaseRepo, Repo
def get_tagged(self, refs=None, repo=None) -> Dict[ObjectID, ObjectID]:
    """Get a dict of peeled values of tags to their original tag shas.

        Args:
          refs: dict of refname -> sha of possible tags; defaults to all
            of the backend's refs.
          repo: optional Repo instance for getting peeled refs; defaults
            to the backend's repo, if available
        Returns: dict of peeled_sha -> tag_sha, where tag_sha is the sha of a
            tag whose peeled value is peeled_sha.
        """
    if not self.has_capability(CAPABILITY_INCLUDE_TAG):
        return {}
    if refs is None:
        refs = self.repo.get_refs()
    if repo is None:
        repo = getattr(self.repo, 'repo', None)
        if repo is None:
            return {}
    tagged = {}
    for name, sha in refs.items():
        peeled_sha = repo.get_peeled(name)
        if peeled_sha != sha:
            tagged[peeled_sha] = sha
    return tagged