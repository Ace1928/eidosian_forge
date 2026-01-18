import json
import os
import sys
from binascii import a2b_base64
from mimetypes import guess_extension
from textwrap import dedent
from traitlets import Set, Unicode
from .base import Preprocessor
def guess_extension_without_jpe(mimetype):
    """
    This function fixes a problem with '.jpe' extensions
    of jpeg images which are then not recognised by latex.
    For any other case, the function works in the same way
    as mimetypes.guess_extension
    """
    ext = guess_extension(mimetype)
    if ext == '.jpe':
        ext = '.jpeg'
    return ext