import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def mreplace(s, chararray, newchararray):
    for a, b in zip(chararray, newchararray):
        s = s.replace(a, b)
    return s