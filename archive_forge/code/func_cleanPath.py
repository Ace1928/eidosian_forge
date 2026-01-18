from __future__ import division         # use "true" division instead of integer division in Python 2 (see PEP 238)
from __future__ import print_function   # use print() as a function in Python 2 (see PEP 3105)
from __future__ import absolute_import  # use absolute imports by default in Python 2 (see PEP 328)
import math
import optparse
import os
import re
import sys
import time
import xml.dom.minidom
from xml.dom import Node, NotFoundErr
from collections import namedtuple, defaultdict
from decimal import Context, Decimal, InvalidOperation, getcontext
import six
from six.moves import range, urllib
from scour.svg_regex import svg_parser
from scour.svg_transform import svg_transform_parser
from scour.yocto_css import parseCssString
from scour import __version__
def cleanPath(element, options):
    """
       Cleans the path string (d attribute) of the element
    """
    global _num_bytes_saved_in_path_data
    global _num_path_segments_removed
    oldPathStr = element.getAttribute('d')
    path = svg_parser.parse(oldPathStr)
    style = _getStyle(element)
    has_round_or_square_linecaps = element.getAttribute('stroke-linecap') in ['round', 'square'] or ('stroke-linecap' in style and style['stroke-linecap'] in ['round', 'square'])
    has_intermediate_markers = element.hasAttribute('marker') or element.hasAttribute('marker-mid') or 'marker' in style or ('marker-mid' in style)
    x = y = 0
    for pathIndex in range(len(path)):
        cmd, data = path[pathIndex]
        i = 0
        if cmd == 'A':
            for i in range(i, len(data), 7):
                data[i + 5] -= x
                data[i + 6] -= y
                x += data[i + 5]
                y += data[i + 6]
            path[pathIndex] = ('a', data)
        elif cmd == 'a':
            x += sum(data[5::7])
            y += sum(data[6::7])
        elif cmd == 'H':
            for i in range(i, len(data)):
                data[i] -= x
                x += data[i]
            path[pathIndex] = ('h', data)
        elif cmd == 'h':
            x += sum(data)
        elif cmd == 'V':
            for i in range(i, len(data)):
                data[i] -= y
                y += data[i]
            path[pathIndex] = ('v', data)
        elif cmd == 'v':
            y += sum(data)
        elif cmd == 'M':
            startx, starty = (data[0], data[1])
            if pathIndex != 0:
                data[0] -= x
                data[1] -= y
            x, y = (startx, starty)
            i = 2
            for i in range(i, len(data), 2):
                data[i] -= x
                data[i + 1] -= y
                x += data[i]
                y += data[i + 1]
            path[pathIndex] = ('m', data)
        elif cmd in ['L', 'T']:
            for i in range(i, len(data), 2):
                data[i] -= x
                data[i + 1] -= y
                x += data[i]
                y += data[i + 1]
            path[pathIndex] = (cmd.lower(), data)
        elif cmd in ['m']:
            if pathIndex == 0:
                startx, starty = (data[0], data[1])
                x, y = (startx, starty)
                i = 2
            else:
                startx = x + data[0]
                starty = y + data[1]
            for i in range(i, len(data), 2):
                x += data[i]
                y += data[i + 1]
        elif cmd in ['l', 't']:
            x += sum(data[0::2])
            y += sum(data[1::2])
        elif cmd in ['S', 'Q']:
            for i in range(i, len(data), 4):
                data[i] -= x
                data[i + 1] -= y
                data[i + 2] -= x
                data[i + 3] -= y
                x += data[i + 2]
                y += data[i + 3]
            path[pathIndex] = (cmd.lower(), data)
        elif cmd in ['s', 'q']:
            x += sum(data[2::4])
            y += sum(data[3::4])
        elif cmd == 'C':
            for i in range(i, len(data), 6):
                data[i] -= x
                data[i + 1] -= y
                data[i + 2] -= x
                data[i + 3] -= y
                data[i + 4] -= x
                data[i + 5] -= y
                x += data[i + 4]
                y += data[i + 5]
            path[pathIndex] = ('c', data)
        elif cmd == 'c':
            x += sum(data[4::6])
            y += sum(data[5::6])
        elif cmd in ['z', 'Z']:
            x, y = (startx, starty)
            path[pathIndex] = ('z', data)
    if not has_round_or_square_linecaps:
        for pathIndex in range(len(path)):
            cmd, data = path[pathIndex]
            i = 0
            if cmd in ['m', 'l', 't']:
                if cmd == 'm':
                    i = 2
                while i < len(data):
                    if data[i] == data[i + 1] == 0:
                        del data[i:i + 2]
                        _num_path_segments_removed += 1
                    else:
                        i += 2
            elif cmd == 'c':
                while i < len(data):
                    if data[i] == data[i + 1] == data[i + 2] == data[i + 3] == data[i + 4] == data[i + 5] == 0:
                        del data[i:i + 6]
                        _num_path_segments_removed += 1
                    else:
                        i += 6
            elif cmd == 'a':
                while i < len(data):
                    if data[i + 5] == data[i + 6] == 0:
                        del data[i:i + 7]
                        _num_path_segments_removed += 1
                    else:
                        i += 7
            elif cmd == 'q':
                while i < len(data):
                    if data[i] == data[i + 1] == data[i + 2] == data[i + 3] == 0:
                        del data[i:i + 4]
                        _num_path_segments_removed += 1
                    else:
                        i += 4
            elif cmd in ['h', 'v']:
                oldLen = len(data)
                path[pathIndex] = (cmd, [coord for coord in data if coord != 0])
                _num_path_segments_removed += len(path[pathIndex][1]) - oldLen
        pathIndex = len(path)
        subpath_needs_anchor = False
        while pathIndex > 1:
            pathIndex -= 1
            cmd, data = path[pathIndex]
            if cmd == 'z':
                next_cmd, next_data = path[pathIndex - 1]
                if next_cmd == 'm' and len(next_data) == 2:
                    del path[pathIndex]
                    _num_path_segments_removed += 1
                else:
                    subpath_needs_anchor = True
            elif cmd == 'm':
                if len(path) - 1 == pathIndex and len(data) == 2:
                    del path[pathIndex]
                    _num_path_segments_removed += 1
                    continue
                if subpath_needs_anchor:
                    subpath_needs_anchor = False
                elif data[0] == data[1] == 0:
                    path[pathIndex] = ('l', data[2:])
                    _num_path_segments_removed += 1
    path = [elem for elem in path if len(elem[1]) > 0 or elem[0] == 'z']
    newPath = [path[0]]
    for cmd, data in path[1:]:
        i = 0
        newData = data
        if cmd == 'c':
            newData = []
            while i < len(data):
                p1x, p1y = (data[i], data[i + 1])
                p2x, p2y = (data[i + 2], data[i + 3])
                dx = data[i + 4]
                dy = data[i + 5]
                foundStraightCurve = False
                if dx == 0:
                    if p1x == 0 and p2x == 0:
                        foundStraightCurve = True
                else:
                    m = dy / dx
                    if p1y == m * p1x and p2y == m * p2x:
                        foundStraightCurve = True
                if foundStraightCurve:
                    if newData:
                        newPath.append((cmd, newData))
                        newData = []
                    newPath.append(('l', [dx, dy]))
                else:
                    newData.extend(data[i:i + 6])
                i += 6
        if newData or cmd == 'z' or cmd == 'Z':
            newPath.append((cmd, newData))
    path = newPath
    prevCmd = ''
    prevData = []
    newPath = []
    for cmd, data in path:
        if prevCmd == '':
            prevCmd = cmd
            prevData = data
        elif cmd != 'm' and (cmd == prevCmd or (cmd == 'l' and prevCmd == 'm')):
            prevData.extend(data)
        else:
            newPath.append((prevCmd, prevData))
            prevCmd = cmd
            prevData = data
    newPath.append((prevCmd, prevData))
    path = newPath
    newPath = []
    for cmd, data in path:
        if cmd == 'l':
            i = 0
            lineTuples = []
            while i < len(data):
                if data[i] == 0:
                    if lineTuples:
                        newPath.append(('l', lineTuples))
                        lineTuples = []
                    newPath.append(('v', [data[i + 1]]))
                    _num_path_segments_removed += 1
                elif data[i + 1] == 0:
                    if lineTuples:
                        newPath.append(('l', lineTuples))
                        lineTuples = []
                    newPath.append(('h', [data[i]]))
                    _num_path_segments_removed += 1
                else:
                    lineTuples.extend(data[i:i + 2])
                i += 2
            if lineTuples:
                newPath.append(('l', lineTuples))
        elif cmd == 'm':
            i = 2
            lineTuples = [data[0], data[1]]
            while i < len(data):
                if data[i] == 0:
                    if lineTuples:
                        newPath.append((cmd, lineTuples))
                        lineTuples = []
                        cmd = 'l'
                    newPath.append(('v', [data[i + 1]]))
                    _num_path_segments_removed += 1
                elif data[i + 1] == 0:
                    if lineTuples:
                        newPath.append((cmd, lineTuples))
                        lineTuples = []
                        cmd = 'l'
                    newPath.append(('h', [data[i]]))
                    _num_path_segments_removed += 1
                else:
                    lineTuples.extend(data[i:i + 2])
                i += 2
            if lineTuples:
                newPath.append((cmd, lineTuples))
        elif cmd == 'c':
            bez_ctl_pt = (0, 0)
            if len(newPath):
                prevCmd, prevData = newPath[-1]
                if prevCmd == 's':
                    bez_ctl_pt = (prevData[-2] - prevData[-4], prevData[-1] - prevData[-3])
            i = 0
            curveTuples = []
            while i < len(data):
                if bez_ctl_pt[0] == data[i] and bez_ctl_pt[1] == data[i + 1]:
                    if curveTuples:
                        newPath.append(('c', curveTuples))
                        curveTuples = []
                    newPath.append(('s', [data[i + 2], data[i + 3], data[i + 4], data[i + 5]]))
                    _num_path_segments_removed += 1
                else:
                    j = 0
                    while j <= 5:
                        curveTuples.append(data[i + j])
                        j += 1
                bez_ctl_pt = (data[i + 4] - data[i + 2], data[i + 5] - data[i + 3])
                i += 6
            if curveTuples:
                newPath.append(('c', curveTuples))
        elif cmd == 'q':
            quad_ctl_pt = (0, 0)
            i = 0
            curveTuples = []
            while i < len(data):
                if quad_ctl_pt[0] == data[i] and quad_ctl_pt[1] == data[i + 1]:
                    if curveTuples:
                        newPath.append(('q', curveTuples))
                        curveTuples = []
                    newPath.append(('t', [data[i + 2], data[i + 3]]))
                    _num_path_segments_removed += 1
                else:
                    j = 0
                    while j <= 3:
                        curveTuples.append(data[i + j])
                        j += 1
                quad_ctl_pt = (data[i + 2] - data[i], data[i + 3] - data[i + 1])
                i += 4
            if curveTuples:
                newPath.append(('q', curveTuples))
        else:
            newPath.append((cmd, data))
    path = newPath
    if not has_intermediate_markers:
        for pathIndex in range(len(path)):
            cmd, data = path[pathIndex]
            if cmd in ['h', 'v'] and len(data) >= 2:
                coordIndex = 0
                while coordIndex + 1 < len(data):
                    if is_same_sign(data[coordIndex], data[coordIndex + 1]):
                        data[coordIndex] += data[coordIndex + 1]
                        del data[coordIndex + 1]
                        _num_path_segments_removed += 1
                    else:
                        coordIndex += 1
            elif cmd == 'l' and len(data) >= 4:
                coordIndex = 0
                while coordIndex + 2 < len(data):
                    if is_same_direction(*data[coordIndex:coordIndex + 4]):
                        data[coordIndex] += data[coordIndex + 2]
                        data[coordIndex + 1] += data[coordIndex + 3]
                        del data[coordIndex + 2]
                        del data[coordIndex + 2]
                        _num_path_segments_removed += 1
                    else:
                        coordIndex += 2
            elif cmd == 'm' and len(data) >= 6:
                coordIndex = 2
                while coordIndex + 2 < len(data):
                    if is_same_direction(*data[coordIndex:coordIndex + 4]):
                        data[coordIndex] += data[coordIndex + 2]
                        data[coordIndex + 1] += data[coordIndex + 3]
                        del data[coordIndex + 2]
                        del data[coordIndex + 2]
                        _num_path_segments_removed += 1
                    else:
                        coordIndex += 2
    prevCmd = ''
    prevData = []
    newPath = [path[0]]
    for cmd, data in path[1:]:
        if prevCmd != '':
            if cmd != prevCmd or cmd == 'm':
                newPath.append((prevCmd, prevData))
                prevCmd = ''
                prevData = []
        if cmd == prevCmd and cmd != 'm':
            prevData.extend(data)
        else:
            prevCmd = cmd
            prevData = data
    if prevCmd != '':
        newPath.append((prevCmd, prevData))
    path = newPath
    newPathStr = serializePath(path, options)
    if len(newPathStr) <= len(oldPathStr):
        _num_bytes_saved_in_path_data += len(oldPathStr) - len(newPathStr)
        element.setAttribute('d', newPathStr)