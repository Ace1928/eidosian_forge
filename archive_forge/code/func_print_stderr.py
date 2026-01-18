import sys, re, curl, exceptions
from the command line first, then standard input.
def print_stderr(arg):
    sys.stderr.write(arg)
    sys.stderr.write('\n')