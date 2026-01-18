import doctest
import collections
@midtop.setter
def midtop(self, value):
    if self._readOnly:
        raise PyRectException('Rect object is read-only')
    _checkForTwoIntOrFloatTuple(value)
    newMidTop, newTop = value
    originalLeft = self._left
    originalTop = self._top
    if self._enableFloat:
        if newMidTop != self._left + self._width / 2.0 or newTop != self._top:
            self._left = newMidTop - self._width / 2.0
            self._top = newTop
            self.callOnChange(originalLeft, originalTop, self._width, self._height)
    elif newMidTop != self._left + self._width // 2 or newTop != self._top:
        self._left = int(newMidTop) - self._width // 2
        self._top = int(newTop)
        self.callOnChange(originalLeft, originalTop, self._width, self._height)