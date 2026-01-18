from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
def read_infile(self, inname):
    """Read the input file"""
    if inname:
        print('Reading input from %s' % inname, file=sys.stderr)
        with open(inname, 'rb') as infile:
            return infile.read()
    print('Reading input from stdin', file=sys.stderr)
    return sys.stdin.read()