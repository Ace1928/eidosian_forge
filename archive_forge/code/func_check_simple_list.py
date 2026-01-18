import sys
import os.path
import re
import urllib.request, urllib.parse, urllib.error
import docutils
from docutils import nodes, utils, writers, languages, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import (unichar2tex, pick_math_environment,
def check_simple_list(self, node):
    """Check for a simple list that can be rendered compactly."""
    visitor = SimpleListChecker(self.document)
    try:
        node.walk(visitor)
    except nodes.NodeFound:
        return False
    else:
        return True