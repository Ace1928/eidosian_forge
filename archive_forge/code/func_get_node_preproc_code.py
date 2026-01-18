import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def get_node_preproc_code(self, node):
    lblstyle = node.attr.get('lblstyle', '')
    shape = node.attr.get('shape', 'ellipse')
    shape = self.shape_map.get(shape, shape)
    label = node.attr.get('texlbl', '')
    style = node.attr.get('style', ' ') or ' '
    if lblstyle:
        if style.strip():
            style += ',' + lblstyle
        else:
            style = lblstyle
    sn = ''
    if self.options.get('styleonly'):
        sn += '\\tikz  \\node [%s] {%s};\n' % (style, label)
    else:
        sn += '\\tikz  \\node [draw,%s,%s] {%s};\n' % (shape, style, label)
    return sn