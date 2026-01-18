import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def get_graph_preproc_code(self, graph):
    lblstyle = graph.attr.get('lblstyle', '')
    text = graph.attr.get('texlbl', '')
    if lblstyle:
        return '  \\tikz \\node[%s] {%s};\n' % (lblstyle, text)
    else:
        return '\\tikz \\node {' + text + '};'