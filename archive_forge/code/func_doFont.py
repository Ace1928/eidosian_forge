import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def doFont(c, s):
    tokens = s.split()
    size = tokens[0]
    n = int(tokens[1])
    tmp = len(size) + len(tokens[1]) + 4
    d = s[tmp:tmp + n]
    didx = len(d) + tmp
    return (didx, (c, size, d))