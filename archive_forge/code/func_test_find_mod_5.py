import shutil
import sys
import tempfile
from pathlib import Path
import IPython.utils.module_paths as mp
def test_find_mod_5():
    """
    Search for a filename with a .pyc extension
    Expected output: TODO: do we exclude or include .pyc files?
    """
    assert mp.find_mod('packpyc') == None