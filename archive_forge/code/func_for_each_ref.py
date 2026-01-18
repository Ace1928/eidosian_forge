import datetime
import fnmatch
import os
import posixpath
import stat
import sys
import time
from collections import namedtuple
from contextlib import closing, contextmanager
from io import BytesIO, RawIOBase
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from .archive import tar_stream
from .client import get_transport_and_path
from .config import Config, ConfigFile, StackedConfig, read_submodules
from .diff_tree import (
from .errors import SendPackError
from .file import ensure_dir_exists
from .graph import can_fast_forward
from .ignore import IgnoreFilterManager
from .index import (
from .object_store import iter_tree_contents, tree_lookup_path
from .objects import (
from .objectspec import (
from .pack import write_pack_from_container, write_pack_index
from .patch import write_tree_diff
from .protocol import ZERO_SHA, Protocol
from .refs import (
from .repo import BaseRepo, Repo
from .server import (
from .server import update_server_info as server_update_server_info
def for_each_ref(repo: Union[Repo, str]='.', pattern: Optional[Union[str, bytes]]=None) -> List[Tuple[bytes, bytes, bytes]]:
    """Iterate over all refs that match the (optional) pattern.

    Args:
      repo: Path to the repository
      pattern: Optional glob (7) patterns to filter the refs with
    Returns:
      List of bytes tuples with: (sha, object_type, ref_name)
    """
    if isinstance(pattern, str):
        pattern = os.fsencode(pattern)
    with open_repo_closing(repo) as r:
        refs = r.get_refs()
    if pattern:
        matching_refs: Dict[bytes, bytes] = {}
        pattern_parts = pattern.split(b'/')
        for ref, sha in refs.items():
            matches = False
            ref_parts = ref.split(b'/')
            if len(ref_parts) > len(pattern_parts):
                continue
            for pat, ref_part in zip(pattern_parts, ref_parts):
                matches = fnmatch.fnmatchcase(ref_part, pat)
                if not matches:
                    break
            if matches:
                matching_refs[ref] = sha
        refs = matching_refs
    ret: List[Tuple[bytes, bytes, bytes]] = [(sha, r.get_object(sha).type_name, ref) for ref, sha in sorted(refs.items(), key=lambda ref_sha: ref_sha[0]) if ref != b'HEAD']
    return ret