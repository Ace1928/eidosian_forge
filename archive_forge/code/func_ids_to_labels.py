import sys
import os
import time
import re
import string
import urllib.request, urllib.parse, urllib.error
from docutils import frontend, nodes, languages, writers, utils, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import pick_math_environment, unichar2tex
def ids_to_labels(self, node, set_anchor=True):
    """Return list of label definitions for all ids of `node`

        If `set_anchor` is True, an anchor is set with \\phantomsection.
        """
    labels = ['\\label{%s}' % id for id in node.get('ids', [])]
    if set_anchor and labels:
        labels.insert(0, '\\phantomsection')
    return labels