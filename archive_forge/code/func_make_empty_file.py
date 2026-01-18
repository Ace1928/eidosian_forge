import shutil
import sys
import tempfile
from pathlib import Path
import IPython.utils.module_paths as mp
def make_empty_file(fname):
    open(fname, 'w', encoding='utf-8').close()