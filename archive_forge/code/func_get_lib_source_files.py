import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def get_lib_source_files(lib):
    filenames = []
    sources = lib[1].get('sources', [])
    sources = [_m for _m in sources if is_string(_m)]
    filenames.extend(sources)
    filenames.extend(get_dependencies(sources))
    depends = lib[1].get('depends', [])
    for d in depends:
        if is_local_src_dir(d):
            filenames.extend(list(general_source_files(d)))
        elif os.path.isfile(d):
            filenames.append(d)
    return filenames