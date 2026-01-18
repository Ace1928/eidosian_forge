import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def parse_drawstring(drawstring):
    """Parse drawstring and returns a list of draw operations"""

    def doeE(c, s):
        """Parse ellipse"""
        tokens = s.split()[0:4]
        if not tokens:
            return None
        points = [float(t) for t in tokens]
        didx = sum((len(t) for t in tokens)) + len(points) + 1
        return (didx, (c, points[0], points[1], points[2], points[3]))

    def doPLB(c, s):
        """Parse polygon, polyline og B-spline"""
        tokens = s.split()
        n = int(tokens[0])
        points = [float(t) for t in tokens[1:n * 2 + 1]]
        didx = sum((len(t) for t in tokens[1:n * 2 + 1])) + n * 2 + 2
        npoints = nsplit(points, 2)
        return (didx, (c, npoints))

    def doCS(c, s):
        """Parse fill or pen color"""
        tokens = s.split()
        n = int(tokens[0])
        tmp = len(tokens[0]) + 3
        d = s[tmp:tmp + n]
        didx = len(d) + tmp + 1
        return (didx, (c, d))

    def doFont(c, s):
        tokens = s.split()
        size = tokens[0]
        n = int(tokens[1])
        tmp = len(size) + len(tokens[1]) + 4
        d = s[tmp:tmp + n]
        didx = len(d) + tmp
        return (didx, (c, size, d))

    def doText(c, s):
        tokens = s.split()
        x, y, j, w = tokens[0:4]
        n = int(tokens[4])
        tmp = sum((len(t) for t in tokens[0:5])) + 7
        text = s[tmp:tmp + n]
        didx = len(text) + tmp
        return (didx, [c, x, y, j, w, text])
    cmdlist = []
    stat = {}
    idx = 0
    s = drawstring.strip().replace('\\', '')
    while idx < len(s) - 1:
        didx = 1
        c = s[idx]
        stat[c] = stat.get(c, 0) + 1
        try:
            if c in ('e', 'E'):
                didx, cmd = doeE(c, s[idx + 1:])
                cmdlist.append(cmd)
            elif c in ('p', 'P', 'L', 'b', 'B'):
                didx, cmd = doPLB(c, s[idx + 1:])
                cmdlist.append(cmd)
            elif c in ('c', 'C', 'S'):
                didx, cmd = doCS(c, s[idx + 1:])
                cmdlist.append(cmd)
            elif c == 'F':
                didx, cmd = doFont(c, s[idx + 1:])
                cmdlist.append(cmd)
            elif c == 'T':
                didx, cmd = doText(c, s[idx + 1:])
                cmdlist.append(cmd)
        except Exception as err:
            log.debug('Failed to parse drawstring %s\n%s', s, err.message)
        idx += didx
    return (cmdlist, stat)