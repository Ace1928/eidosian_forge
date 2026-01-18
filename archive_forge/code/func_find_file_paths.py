import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
def find_file_paths(dir_paths: List[str], files_to_exclude: Set[str]) -> Set[str]:
    """
    When given a path to a directory, returns the paths to the relevant files within it.

    This function does NOT recursive traverse to subdirectories.
    """
    paths: Set[str] = set()
    for dir_path in dir_paths:
        all_files = os.listdir(dir_path)
        python_files = {fname for fname in all_files if '.py' == fname[-3:]}
        filter_files = {fname for fname in python_files if fname not in files_to_exclude}
        paths.update({os.path.join(dir_path, fname) for fname in filter_files})
    return paths