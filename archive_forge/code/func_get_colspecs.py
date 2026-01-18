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
def get_colspecs(self, node):
    """Return column specification for longtable.

        Assumes reST line length being 80 characters.
        Table width is hairy.

        === ===
        ABC DEF
        === ===

        usually gets to narrow, therefore we add 1 (fiddlefactor).
        """
    bar = self.get_vertical_bar()
    self._rowspan = [0] * len(self._col_specs)
    self._col_width = []
    if self.colwidths_auto:
        latex_table_spec = (bar + 'l') * len(self._col_specs)
        return latex_table_spec + bar
    width = 80
    total_width = 0.0
    for node in self._col_specs:
        colwidth = float(node['colwidth'] + 1) / width
        total_width += colwidth
    factor = 0.93
    if total_width > 1.0:
        factor /= total_width
    latex_table_spec = ''
    for node in self._col_specs:
        colwidth = factor * float(node['colwidth'] + 1) / width
        self._col_width.append(colwidth + 0.005)
        latex_table_spec += '%sp{%.3f\\DUtablewidth}' % (bar, colwidth + 0.005)
    return latex_table_spec + bar