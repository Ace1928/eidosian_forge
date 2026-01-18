import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def doPLB(c, s):
    """Parse polygon, polyline og B-spline"""
    tokens = s.split()
    n = int(tokens[0])
    points = [float(t) for t in tokens[1:n * 2 + 1]]
    didx = sum((len(t) for t in tokens[1:n * 2 + 1])) + n * 2 + 2
    npoints = nsplit(points, 2)
    return (didx, (c, npoints))