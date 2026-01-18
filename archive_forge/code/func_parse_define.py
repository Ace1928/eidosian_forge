import io
import logging
import os
from shlex import split as shsplit
import sys
import numpy
def parse_define(define):
    index = define.find('=')
    if index < 0:
        return (define, None)
    else:
        return (define[:index], define[index + 1:])