import json
from pathlib import Path
from itertools import chain
from jedi import debug
from jedi.api.environment import get_cached_default_environment, create_environment
from jedi.api.exceptions import WrongVersion
from jedi.api.completion import search_in_module
from jedi.api.helpers import split_search_string, get_module_names
from jedi.inference.imports import load_module_from_path, \
from jedi.inference.sys_path import discover_buildout_paths
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.inference.references import recurse_find_python_folders_and_files, search_in_file_ios
from jedi.file_io import FolderIO
def get_default_project(path=None):
    """
    If a project is not defined by the user, Jedi tries to define a project by
    itself as well as possible. Jedi traverses folders until it finds one of
    the following:

    1. A ``.jedi/config.json``
    2. One of the following files: ``setup.py``, ``.git``, ``.hg``,
       ``requirements.txt`` and ``MANIFEST.in``.
    """
    if path is None:
        path = Path.cwd()
    elif isinstance(path, str):
        path = Path(path)
    check = path.absolute()
    probable_path = None
    first_no_init_file = None
    for dir in chain([check], check.parents):
        try:
            return Project.load(dir)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            pass
        except NotADirectoryError:
            continue
        if first_no_init_file is None:
            if dir.joinpath('__init__.py').exists():
                continue
            elif not dir.is_file():
                first_no_init_file = dir
        if _is_django_path(dir):
            project = Project(dir)
            project._django = True
            return project
        if probable_path is None and _is_potential_project(dir):
            probable_path = dir
    if probable_path is not None:
        return Project(probable_path)
    if first_no_init_file is not None:
        return Project(first_no_init_file)
    curdir = path if path.is_dir() else path.parent
    return Project(curdir)