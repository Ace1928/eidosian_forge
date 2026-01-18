import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def do_nodes(self):
    s = ''
    for node in self.nodes:
        self.currentnode = node
        psshadeoption = getattr(node, 'psshadeoption', '')
        psshape = getattr(node, 'psshape', '')
        if len(psshape) == 0:
            shape = getattr(node, 'shape', 'ellipse')
            psshape = 'psframebox'
            if shape == 'circle':
                psshape = 'pscirclebox'
            if shape == 'ellipse':
                psshape = 'psovalbox'
            if shape == 'triangle':
                psshape = 'pstribox'
        width = getattr(node, 'width', '1')
        height = getattr(node, 'height', '1')
        psbox = getattr(node, 'psbox', 'false')
        color = getattr(node, 'color', '')
        fillcolor = getattr(node, 'fillcolor', '')
        if len(color) > 0:
            psshadeoption = 'linecolor=' + color + ',' + psshadeoption
        if len(fillcolor) > 0:
            psshadeoption = 'fillcolor=' + fillcolor + ',' + psshadeoption
        style = getattr(node, 'style', '')
        if len(style) > 0:
            if style == 'dotted':
                psshadeoption = 'linestyle=dotted,' + psshadeoption
            if style == 'dashed':
                psshadeoption = 'linestyle=dashed,' + psshadeoption
            if style == 'solid':
                psshadeoption = 'linestyle=solid,' + psshadeoption
            if style == 'bold':
                psshadeoption = 'linewidth=2pt,' + psshadeoption
        pos = getattr(node, 'pos')
        if not pos:
            continue
        x, y = pos.split(',')
        label = self.get_label(node)
        pos = '%sbp,%sbp' % (smart_float(x), smart_float(y))
        sn = ''
        sn += self.output_node_comment(node)
        sn += self.start_node(node)
        if psbox == 'false':
            sn += '\\rput(%s){\\rnode{%s}{\\%s[%s]{%s}}}\n' % (pos, tikzify(node.name), psshape, psshadeoption, label)
        else:
            sn += '\\rput(%s){\\rnode{%s}{\\%s[%s]{\\parbox[c][%sin][c]{%sin}{\\centering %s}}}}\n' % (pos, tikzify(node.name), psshape, psshadeoption, height, width, label)
        sn += self.end_node(node)
        s += sn
    self.body += s