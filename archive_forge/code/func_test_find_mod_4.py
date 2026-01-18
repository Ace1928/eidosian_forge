import shutil
import sys
import tempfile
from pathlib import Path
import IPython.utils.module_paths as mp
def test_find_mod_4():
    """
    Search for a filename without its .py extension
    Expected output: full path with .py extension
    """
    modpath = TMP_TEST_DIR / 'pack.py'
    assert Path(mp.find_mod('pack')) == modpath