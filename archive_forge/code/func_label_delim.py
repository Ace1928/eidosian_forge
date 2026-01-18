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
def label_delim(self, node, bracket, superscript):
    if isinstance(node.parent, nodes.footnote):
        raise nodes.SkipNode
    else:
        assert isinstance(node.parent, nodes.citation)
        if not self._use_latex_citations:
            self.out.append(bracket)