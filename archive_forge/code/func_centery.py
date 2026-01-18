import doctest
import collections
@centery.setter
def centery(self, newCentery):
    if self._readOnly:
        raise PyRectException('Rect object is read-only')
    _checkForIntOrFloat(newCentery)
    originalTop = self._top
    if self._enableFloat:
        if newCentery != self._top + self._height / 2.0:
            self._top = newCentery - self._height / 2.0
            self.callOnChange(self._left, originalTop, self._width, self._height)
    elif newCentery != self._top + self._height // 2:
        self._top = int(newCentery) - self._height // 2
        self.callOnChange(self._left, originalTop, self._width, self._height)