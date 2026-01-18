import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def doCS(c, s):
    """Parse fill or pen color"""
    tokens = s.split()
    n = int(tokens[0])
    tmp = len(tokens[0]) + 3
    d = s[tmp:tmp + n]
    didx = len(d) + tmp + 1
    return (didx, (c, d))