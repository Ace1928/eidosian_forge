import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def recurse_find_python_files(folder_io, except_paths=()):
    for folder_io, file_io in recurse_find_python_folders_and_files(folder_io, except_paths):
        if file_io is not None:
            yield file_io