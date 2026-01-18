from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def is_string_literal(line):
    if line[0] in 'uUbB':
        line = line[1:]
    if line and line[0] in 'rR':
        line = line[1:]
    return line and (line[0] == '"' or line[0] == "'")