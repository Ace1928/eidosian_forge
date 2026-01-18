import glob
import importlib
import inspect
import logging
import os
import re
import sys
from typing import Iterable, List, Optional, Tuple, Union
def adjust_linked_external_docs(source_link: str, target_link: str, browse_folder: Union[str, Iterable[str]], file_extensions: Iterable[str]=('.rst', '.py'), version_digits: int=2) -> None:
    """Adjust the linked external docs to be local.

    Args:
        source_link: the link to be replaced
        target_link: the link to be replaced, if ``{package.version}`` is included it will be replaced accordingly
        browse_folder: the location of the browsable folder
        file_extensions: what kind of files shall be scanned
        version_digits: for semantic versioning, how many digits to be considered

    Examples:
        >>> adjust_linked_external_docs(
        ...     "https://numpy.org/doc/stable/",
        ...     "https://numpy.org/doc/{numpy.__version__}/",
        ...     "docs/source",
        ... )

    """
    list_files = []
    if isinstance(browse_folder, str):
        browse_folder = [browse_folder]
    for folder in browse_folder:
        for ext in file_extensions:
            list_files += glob.glob(os.path.join(folder, '**', f'*{ext}'), recursive=True)
    if not list_files:
        logging.warning(f'No files were listed in folder "{browse_folder}" and pattern "{file_extensions}"')
        return
    pkg_ver_all = re.findall('{(.+)}', target_link)
    for pkg_ver in pkg_ver_all:
        target_link = _update_link_based_imported_package(target_link, pkg_ver, version_digits)
    for fpath in set(list_files):
        with open(fpath, encoding='UTF-8') as fopen:
            lines = fopen.readlines()
        found, skip = (False, False)
        for i, ln in enumerate(lines):
            if f'{adjust_linked_external_docs.__name__}(' in ln:
                skip = True
            if not skip and source_link in ln:
                lines[i] = ln.replace(source_link, target_link)
                found = True
            if skip and ')' in ln:
                skip = False
        if not found:
            continue
        logging.debug(f'links adjusting in {fpath}: "{source_link}" -> "{target_link}"')
        with open(fpath, 'w', encoding='UTF-8') as fw:
            fw.writelines(lines)