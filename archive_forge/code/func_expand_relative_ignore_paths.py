import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def expand_relative_ignore_paths(folder_io, relative_paths):
    curr_path = folder_io.path
    return {os.path.join(curr_path, p[1]) for p in relative_paths if curr_path.startswith(p[0])}