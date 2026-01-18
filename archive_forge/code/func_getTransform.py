import copy
from enum import IntEnum
from functools import reduce
from math import radians
import itertools
from collections import defaultdict, namedtuple
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.misc.arrayTools import quantizeRect
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform, Identity
from fontTools.misc.textTools import bytesjoin, pad, safeEval
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.transformPen import TransformPen
from .otBase import (
from fontTools.feaLib.lookupDebugInfo import LookupDebugInfo, LOOKUP_DEBUG_INFO_KEY
import logging
import struct
from typing import TYPE_CHECKING, Iterator, List, Optional, Set
def getTransform(self) -> Transform:
    if self.Format == PaintFormat.PaintTransform:
        t = self.Transform
        return Transform(t.xx, t.yx, t.xy, t.yy, t.dx, t.dy)
    elif self.Format == PaintFormat.PaintTranslate:
        return Identity.translate(self.dx, self.dy)
    elif self.Format == PaintFormat.PaintScale:
        return Identity.scale(self.scaleX, self.scaleY)
    elif self.Format == PaintFormat.PaintScaleAroundCenter:
        return Identity.translate(self.centerX, self.centerY).scale(self.scaleX, self.scaleY).translate(-self.centerX, -self.centerY)
    elif self.Format == PaintFormat.PaintScaleUniform:
        return Identity.scale(self.scale)
    elif self.Format == PaintFormat.PaintScaleUniformAroundCenter:
        return Identity.translate(self.centerX, self.centerY).scale(self.scale).translate(-self.centerX, -self.centerY)
    elif self.Format == PaintFormat.PaintRotate:
        return Identity.rotate(radians(self.angle))
    elif self.Format == PaintFormat.PaintRotateAroundCenter:
        return Identity.translate(self.centerX, self.centerY).rotate(radians(self.angle)).translate(-self.centerX, -self.centerY)
    elif self.Format == PaintFormat.PaintSkew:
        return Identity.skew(radians(-self.xSkewAngle), radians(self.ySkewAngle))
    elif self.Format == PaintFormat.PaintSkewAroundCenter:
        return Identity.translate(self.centerX, self.centerY).skew(radians(-self.xSkewAngle), radians(self.ySkewAngle)).translate(-self.centerX, -self.centerY)
    if PaintFormat(self.Format).is_variable():
        raise NotImplementedError(f'Variable Paints not supported: {self.Format}')
    return Identity