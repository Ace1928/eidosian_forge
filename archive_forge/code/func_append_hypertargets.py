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
def append_hypertargets(self, node):
    """Append hypertargets for all ids of `node`"""
    self.out.append('%\n'.join(['\\raisebox{1em}{\\hypertarget{%s}{}}' % id for id in node['ids']]))