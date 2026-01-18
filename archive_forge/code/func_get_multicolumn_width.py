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
def get_multicolumn_width(self, start, len_):
    """Return sum of columnwidths for multicell."""
    try:
        mc_width = sum([width for width in [self._col_width[start + co] for co in range(len_)]])
        return 'p{%.2f\\DUtablewidth}' % mc_width
    except IndexError:
        return 'l'