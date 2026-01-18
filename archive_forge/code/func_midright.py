import doctest
import collections
@midright.setter
def midright(self, value):
    if self._readOnly:
        raise PyRectException('Rect object is read-only')
    _checkForTwoIntOrFloatTuple(value)
    newRight, newMidRight = value
    originalLeft = self._left
    originalTop = self._top
    if self._enableFloat:
        if newRight != self._left + self._width or newMidRight != self._top + self._height / 2.0:
            self._left = newRight - self._width
            self._top = newMidRight - self._height / 2.0
            self.callOnChange(originalLeft, originalTop, self._width, self._height)
    elif newRight != self._left + self._width or newMidRight != self._top + self._height // 2:
        self._left = int(newRight) - self._width
        self._top = int(newMidRight) - self._height // 2
        self.callOnChange(originalLeft, originalTop, self._width, self._height)