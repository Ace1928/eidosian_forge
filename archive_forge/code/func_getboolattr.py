import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def getboolattr(item, key, default):
    if str(getattr(item, key, '')).lower() == 'true':
        return True
    else:
        return False