import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
@contextlib.contextmanager
def tempdir(*args, **kwargs):
    tmpdir = tempfile.mkdtemp(*args, **kwargs)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)