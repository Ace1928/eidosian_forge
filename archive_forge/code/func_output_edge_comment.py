import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def output_edge_comment(self, edge):
    src = edge.get_source()
    dst = edge.get_destination()
    if self.directedgraph:
        edge = '->'
    else:
        edge = '--'
    return '  %% Edge: %s %s %s\n' % (src, edge, dst)