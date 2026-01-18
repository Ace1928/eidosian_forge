import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def is_multiline_label(drawobject):
    if getattr(drawobject, 'texlbl', None):
        return False
    label = getattr(drawobject, 'label', '')
    return any((x in label for x in ['\\n', '\\l', '\\r']))