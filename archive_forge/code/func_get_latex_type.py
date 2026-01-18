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
def get_latex_type(self):
    if self._latex_type == 'longtable' and (not self.caption):
        return 'longtable*'
    return self._latex_type