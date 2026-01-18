import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def get_edge_points(self, edge):
    pos = edge.attr.get('pos')
    if pos:
        segments = pos.split(';')
    else:
        return []
    return_segments = []
    for pos in segments:
        points = pos.split(' ')
        arrow_style = '--'
        i = 0
        if points[i].startswith('s'):
            p = points[0].split(',')
            tmp = '%s,%s' % (p[1], p[2])
            if points[1].startswith('e'):
                points[2] = tmp
            else:
                points[1] = tmp
            del points[0]
            arrow_style = '<-'
            i += 1
        if points[0].startswith('e'):
            p = points[0].split(',')
            points.pop()
            points.append('%s,%s' % (p[1], p[2]))
            del points[0]
            arrow_style = '->'
            i += 1
        if i > 1:
            arrow_style = '<->'
        arrow_style = self.get_output_arrow_styles(arrow_style, edge)
        return_segments.append((arrow_style, points))
    return return_segments