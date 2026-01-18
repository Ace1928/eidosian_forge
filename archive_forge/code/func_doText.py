import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def doText(c, s):
    tokens = s.split()
    x, y, j, w = tokens[0:4]
    n = int(tokens[4])
    tmp = sum((len(t) for t in tokens[0:5])) + 7
    text = s[tmp:tmp + n]
    didx = len(text) + tmp
    return (didx, [c, x, y, j, w, text])