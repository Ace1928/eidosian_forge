import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def generateDrawSpecs(self, p):
    """
        Calls tickValues() and tickStrings() to determine where and how ticks should
        be drawn, then generates from this a set of drawing commands to be
        interpreted by drawPicture().
        """
    profiler = debug.Profiler()
    if self.style['tickFont'] is not None:
        p.setFont(self.style['tickFont'])
    bounds = self.mapRectFromParent(self.geometry())
    linkedView = self.linkedView()
    if linkedView is None or self.grid is False:
        tickBounds = bounds
    else:
        tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())
    left_offset = -1.0
    right_offset = 1.0
    top_offset = -1.0
    bottom_offset = 1.0
    if self.orientation == 'left':
        span = (bounds.topRight() + Point(left_offset, top_offset), bounds.bottomRight() + Point(left_offset, bottom_offset))
        tickStart = tickBounds.right()
        tickStop = bounds.right()
        tickDir = -1
        axis = 0
    elif self.orientation == 'right':
        span = (bounds.topLeft() + Point(right_offset, top_offset), bounds.bottomLeft() + Point(right_offset, bottom_offset))
        tickStart = tickBounds.left()
        tickStop = bounds.left()
        tickDir = 1
        axis = 0
    elif self.orientation == 'top':
        span = (bounds.bottomLeft() + Point(left_offset, top_offset), bounds.bottomRight() + Point(right_offset, top_offset))
        tickStart = tickBounds.bottom()
        tickStop = bounds.bottom()
        tickDir = -1
        axis = 1
    elif self.orientation == 'bottom':
        span = (bounds.topLeft() + Point(left_offset, bottom_offset), bounds.topRight() + Point(right_offset, bottom_offset))
        tickStart = tickBounds.top()
        tickStop = bounds.top()
        tickDir = 1
        axis = 1
    else:
        raise ValueError("self.orientation must be in ('left', 'right', 'top', 'bottom')")
    points = list(map(self.mapToDevice, span))
    if None in points:
        return
    lengthInPixels = Point(points[1] - points[0]).length()
    if lengthInPixels == 0:
        return
    if self._tickLevels is None:
        tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
        tickStrings = None
    else:
        tickLevels = []
        tickStrings = []
        for level in self._tickLevels:
            values = []
            strings = []
            tickLevels.append((None, values))
            tickStrings.append(strings)
            for val, strn in level:
                values.append(val)
                strings.append(strn)
    dif = self.range[1] - self.range[0]
    if dif == 0:
        xScale = 1
        offset = 0
    elif axis == 0:
        xScale = -bounds.height() / dif
        offset = self.range[0] * xScale - bounds.height()
    else:
        xScale = bounds.width() / dif
        offset = self.range[0] * xScale
    xRange = [x * xScale - offset for x in self.range]
    xMin = min(xRange)
    xMax = max(xRange)
    profiler('init')
    tickPositions = []
    tickSpecs = []
    for i in range(len(tickLevels)):
        tickPositions.append([])
        ticks = tickLevels[i][1]
        tickLength = self.style['tickLength'] / (i * 0.5 + 1.0)
        lineAlpha = self.style['tickAlpha']
        if lineAlpha is None:
            lineAlpha = 255 / (i + 1)
            if self.grid is not False:
                lineAlpha *= self.grid / 255.0 * fn.clip_scalar(0.05 * lengthInPixels / (len(ticks) + 1), 0.0, 1.0)
        elif isinstance(lineAlpha, float):
            lineAlpha *= 255
            lineAlpha = max(0, int(round(lineAlpha)))
            lineAlpha = min(255, int(round(lineAlpha)))
        elif isinstance(lineAlpha, int):
            if lineAlpha > 255 or lineAlpha < 0:
                raise ValueError('lineAlpha should be [0..255]')
        else:
            raise TypeError('Line Alpha should be of type None, float or int')
        tickPen = self.tickPen()
        if tickPen.brush().style() == QtCore.Qt.BrushStyle.SolidPattern:
            tickPen = QtGui.QPen(tickPen)
            color = QtGui.QColor(tickPen.color())
            color.setAlpha(int(lineAlpha))
            tickPen.setColor(color)
        for v in ticks:
            x = v * xScale - offset
            if x < xMin or x > xMax:
                tickPositions[i].append(None)
                continue
            tickPositions[i].append(x)
            p1 = [x, x]
            p2 = [x, x]
            p1[axis] = tickStart
            p2[axis] = tickStop
            if self.grid is False:
                p2[axis] += tickLength * tickDir
            tickSpecs.append((tickPen, Point(p1), Point(p2)))
    profiler('compute ticks')
    if self.style['stopAxisAtTick'][0] is True:
        minTickPosition = min(map(min, tickPositions))
        if axis == 0:
            stop = max(span[0].y(), minTickPosition)
            span[0].setY(stop)
        else:
            stop = max(span[0].x(), minTickPosition)
            span[0].setX(stop)
    if self.style['stopAxisAtTick'][1] is True:
        maxTickPosition = max(map(max, tickPositions))
        if axis == 0:
            stop = min(span[1].y(), maxTickPosition)
            span[1].setY(stop)
        else:
            stop = min(span[1].x(), maxTickPosition)
            span[1].setX(stop)
    axisSpec = (self.pen(), span[0], span[1])
    textOffset = self.style['tickTextOffset'][axis]
    textSize2 = 0
    lastTextSize2 = 0
    textRects = []
    textSpecs = []
    if not self.style['showValues']:
        return (axisSpec, tickSpecs, textSpecs)
    for i in range(min(len(tickLevels), self.style['maxTextLevel'] + 1)):
        if tickStrings is None:
            spacing, values = tickLevels[i]
            strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
        else:
            strings = tickStrings[i]
        if len(strings) == 0:
            continue
        for j in range(len(strings)):
            if tickPositions[i][j] is None:
                strings[j] = None
        rects = []
        for s in strings:
            if s is None:
                rects.append(None)
            else:
                br = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignmentFlag.AlignCenter, s)
                br.setHeight(br.height() * 0.8)
                rects.append(br)
                textRects.append(rects[-1])
        if len(textRects) > 0:
            if axis == 0:
                textSize = np.sum([r.height() for r in textRects])
                textSize2 = np.max([r.width() for r in textRects])
            else:
                textSize = np.sum([r.width() for r in textRects])
                textSize2 = np.max([r.height() for r in textRects])
        else:
            textSize = 0
            textSize2 = 0
        if i > 0:
            textFillRatio = float(textSize) / lengthInPixels
            finished = False
            for nTexts, limit in self.style['textFillLimits']:
                if len(textSpecs) >= nTexts and textFillRatio >= limit:
                    finished = True
                    break
            if finished:
                break
        lastTextSize2 = textSize2
        for j in range(len(strings)):
            vstr = strings[j]
            if vstr is None:
                continue
            x = tickPositions[i][j]
            textRect = rects[j]
            height = textRect.height()
            width = textRect.width()
            offset = max(0, self.style['tickLength']) + textOffset
            rect = QtCore.QRectF()
            if self.orientation == 'left':
                alignFlags = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                rect = QtCore.QRectF(tickStop - offset - width, x - height / 2, width, height)
            elif self.orientation == 'right':
                alignFlags = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
                rect = QtCore.QRectF(tickStop + offset, x - height / 2, width, height)
            elif self.orientation == 'top':
                alignFlags = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignBottom
                rect = QtCore.QRectF(x - width / 2.0, tickStop - offset - height, width, height)
            elif self.orientation == 'bottom':
                alignFlags = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop
                rect = QtCore.QRectF(x - width / 2.0, tickStop + offset, width, height)
            textFlags = alignFlags | QtCore.Qt.TextFlag.TextDontClip
            br = self.boundingRect()
            if not br.contains(rect):
                continue
            textSpecs.append((rect, textFlags, vstr))
    profiler('compute text')
    self._updateMaxTextSize(lastTextSize2)
    return (axisSpec, tickSpecs, textSpecs)