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
def latex_to_html(s, alt='image'):
    """Render LaTeX to HTML with embedded PNG data using data URIs.

    Parameters
    ----------
    s : str
        The raw string containing valid inline LateX.
    alt : str
        The alt text to use for the HTML.
    """
    base64_data = latex_to_png(s, encode=True).decode('ascii')
    if base64_data:
        return _data_uri_template_png % (base64_data, alt)