from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asBytes
from reportlab.platypus.paraparser import _num as paraparser_num
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isColor, isString, isColorOrNone, isNumber, isBoxAnchor
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import toColor
from reportlab.graphics.shapes import Group, Rect
def recalc(self):
    if not self._recalc:
        return
    data = self._value
    size = self._size
    encoding = self._encoding
    e = pylibdmtx.encode(data, size=size, scheme=encoding)
    iW = e.width
    iH = e.height
    p = e.pixels
    iCellSize = 5
    bpp = 3
    rowLen = iW * bpp
    cellLen = iCellSize * bpp
    assert len(p) // rowLen == iH
    matrix = list(filter(None, (''.join(('x' if p[j:j + bpp] != b'\xff\xff\xff' else ' ' for j in range(i, i + rowLen, cellLen))).strip() for i in range(0, iH * rowLen, rowLen * iCellSize))))
    self._nRows = len(matrix)
    self._nCols = len(matrix[-1])
    self._matrix = '\n'.join(matrix)
    cellWidth = self._cellSize
    if cellWidth:
        cellWidth = cellWidth.split('x')
        if len(cellWidth) > 2:
            raise ValueError('cellSize needs to be distance x distance not %r' % self._cellSize)
        elif len(cellWidth) == 2:
            cellWidth, cellHeight = cellWidth
        else:
            cellWidth = cellHeight = cellWidth[0]
        cellWidth = _numConv(cellWidth)
        cellHeight = _numConv(cellHeight)
    else:
        cellWidth = cellHeight = iCellSize
    self._cellWidth = cellWidth
    self._cellHeight = cellHeight
    self._recalc = False
    self._bord = max(self.border, cellWidth, cellHeight)
    self._width = cellWidth * self._nCols + 2 * self._bord
    self._height = cellHeight * self._nRows + 2 * self._bord