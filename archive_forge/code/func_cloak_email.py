import sys
import os.path
import re
import urllib.request, urllib.parse, urllib.error
import docutils
from docutils import nodes, utils, writers, languages, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import (unichar2tex, pick_math_environment,
def cloak_email(self, addr):
    """Try to hide the link text of a email link from harversters."""
    addr = addr.replace('&#64;', '<span>&#64;</span>')
    addr = addr.replace('.', '<span>&#46;</span>')
    return addr