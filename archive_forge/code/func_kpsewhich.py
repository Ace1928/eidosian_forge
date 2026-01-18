from io import BytesIO, open
import os
import tempfile
import shutil
import subprocess
from base64 import encodebytes
import textwrap
from pathlib import Path
from IPython.utils.process import find_cmd, FindCmdError
from traitlets.config import get_config
from traitlets.config.configurable import SingletonConfigurable
from traitlets import List, Bool, Unicode
from IPython.utils.py3compat import cast_unicode
def kpsewhich(filename):
    """Invoke kpsewhich command with an argument `filename`."""
    try:
        find_cmd('kpsewhich')
        proc = subprocess.Popen(['kpsewhich', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        return stdout.strip().decode('utf8', 'replace')
    except FindCmdError:
        pass