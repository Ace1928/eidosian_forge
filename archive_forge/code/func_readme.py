from __future__ import print_function
import sys
from boto import __version__
def readme():
    with open('README.rst') as f:
        return f.read()