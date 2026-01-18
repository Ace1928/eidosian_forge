import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def convert_color(self, drawopcolor, pgf=False):
    """Convert color to a format usable by LaTeX and XColor"""
    if drawopcolor.startswith('#'):
        t = list(chunks(drawopcolor[1:], 2))
        if len(t) > 6:
            t = t[0:3]
        rgb = [round(int(n, 16) / 255.0, 2) for n in t]
        if pgf:
            colstr = '{rgb}{%s,%s,%s}' % tuple(rgb[0:3])
            opacity = '1'
            if len(rgb) == 4:
                opacity = rgb[3]
            return (colstr, opacity)
        else:
            return '[rgb]{%s,%s,%s}' % tuple(rgb[0:3])
    elif len(drawopcolor.split(' ')) == 3 or len(drawopcolor.split(',')) == 3:
        hsb = drawopcolor.split(',')
        if not len(hsb) == 3:
            hsb = drawopcolor.split(' ')
        if pgf:
            return '{hsb}{%s,%s,%s}' % tuple(hsb)
        else:
            return '[hsb]{%s,%s,%s}' % tuple(hsb)
    else:
        drawopcolor = drawopcolor.replace('grey', 'gray')
        drawopcolor = drawopcolor.replace('_', '')
        drawopcolor = drawopcolor.replace(' ', '')
        return drawopcolor