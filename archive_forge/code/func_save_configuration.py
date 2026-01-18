import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from . import DistlibException
from .compat import (HTTPBasicAuthHandler, Request, HTTPPasswordMgr,
from .util import zip_dir, ServerProxy
def save_configuration(self):
    """
        Save the PyPI access configuration. You must have set ``username`` and
        ``password`` attributes before calling this method.
        """
    self.check_credentials()
    from .util import _store_pypirc
    _store_pypirc(self)