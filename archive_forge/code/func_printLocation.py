import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def printLocation(depth=1):
    if sys._getframe(depth).f_locals.get('__name__') == '__main__':
        outDir = outputfile('')
        if outDir != _OUTDIR:
            print('Logs and output files written to folder "%s"' % outDir)