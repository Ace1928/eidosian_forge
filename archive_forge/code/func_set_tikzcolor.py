import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def set_tikzcolor(self, color, colorname):
    res = self.convert_color(color, True)
    if len(res) == 2:
        ccolor, opacity = res
        if not opacity == '1':
            log.warning('Opacity not supported yet: %s', res)
    else:
        ccolor = res
    s = ''
    if ccolor.startswith('{'):
        s += '  \\definecolor{%s}%s;\n' % (colorname, ccolor)
        cname = colorname
    else:
        cname = color
    return (s, cname)