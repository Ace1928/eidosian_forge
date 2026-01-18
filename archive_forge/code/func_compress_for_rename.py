import functools
import os
import sys
import sysconfig
from importlib.util import cache_from_source
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple
from pip._internal.exceptions import UninstallationError
from pip._internal.locations import get_bin_prefix, get_bin_user
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.egg_link import egg_link_path_from_location
from pip._internal.utils.logging import getLogger, indent_log
from pip._internal.utils.misc import ask, normalize_path, renames, rmtree
from pip._internal.utils.temp_dir import AdjacentTempDirectory, TempDirectory
from pip._internal.utils.virtualenv import running_under_virtualenv
def compress_for_rename(paths: Iterable[str]) -> Set[str]:
    """Returns a set containing the paths that need to be renamed.

    This set may include directories when the original sequence of paths
    included every file on disk.
    """
    case_map = {os.path.normcase(p): p for p in paths}
    remaining = set(case_map)
    unchecked = sorted({os.path.split(p)[0] for p in case_map.values()}, key=len)
    wildcards: Set[str] = set()

    def norm_join(*a: str) -> str:
        return os.path.normcase(os.path.join(*a))
    for root in unchecked:
        if any((os.path.normcase(root).startswith(w) for w in wildcards)):
            continue
        all_files: Set[str] = set()
        all_subdirs: Set[str] = set()
        for dirname, subdirs, files in os.walk(root):
            all_subdirs.update((norm_join(root, dirname, d) for d in subdirs))
            all_files.update((norm_join(root, dirname, f) for f in files))
        if not all_files - remaining:
            remaining.difference_update(all_files)
            wildcards.add(root + os.sep)
    return set(map(case_map.__getitem__, remaining)) | wildcards