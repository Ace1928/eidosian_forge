import io
import logging
import os
from shlex import split as shsplit
import sys
import numpy
def fmt_define(define):
    name, value = define
    if value is None:
        return '-D' + name
    else:
        return '-D' + name + '=' + value