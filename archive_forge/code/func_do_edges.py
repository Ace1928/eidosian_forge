import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def do_edges(self):
    s = ''
    for edge in self.edges:
        s += self.draw_edge(edge)
    self.body += s