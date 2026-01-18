import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def output_node_comment(self, node):
    return ''