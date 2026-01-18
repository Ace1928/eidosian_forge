import doctest
import collections
@bottomright.setter
def bottomright(self, value):
    if self._readOnly:
        raise PyRectException('Rect object is read-only')
    _checkForTwoIntOrFloatTuple(value)
    newRight, newBottom = value
    if newBottom != self._top + self._height or newRight != self._left + self._width:
        originalLeft = self._left
        originalTop = self._top
        if self._enableFloat:
            self._left = newRight - self._width
            self._top = newBottom - self._height
        else:
            self._left = int(newRight) - self._width
            self._top = int(newBottom) - self._height
        self.callOnChange(originalLeft, originalTop, self._width, self._height)