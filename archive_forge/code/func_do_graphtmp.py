import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def do_graphtmp(self):
    self.pencolor = ''
    self.fillcolor = ''
    self.color = ''
    self.body += '{\n'
    DotConvBase.do_graph(self)
    self.body += '}\n'