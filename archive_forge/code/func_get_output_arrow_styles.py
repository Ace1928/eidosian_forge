import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def get_output_arrow_styles(self, arrow_style, edge):
    dot_arrow_head = edge.attr.get('arrowhead')
    dot_arrow_tail = edge.attr.get('arrowtail')
    output_arrow_style = arrow_style
    if dot_arrow_head:
        pgf_arrow_head = self.arrows_map_210.get(dot_arrow_head)
        if pgf_arrow_head:
            output_arrow_style = output_arrow_style.replace('>', pgf_arrow_head)
    if dot_arrow_tail:
        pgf_arrow_tail = self.arrows_map_210.get(dot_arrow_tail)
        if pgf_arrow_tail:
            output_arrow_style = output_arrow_style.replace('<', pgf_arrow_tail)
    return output_arrow_style