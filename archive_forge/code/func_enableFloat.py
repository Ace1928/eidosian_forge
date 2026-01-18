import doctest
import collections
@enableFloat.setter
def enableFloat(self, value):
    if not isinstance(value, bool):
        raise PyRectException('enableFloat must be set to a bool value')
    self._enableFloat = value
    if self._enableFloat:
        self._left = float(self._left)
        self._top = float(self._top)
        self._width = float(self._width)
        self._height = float(self._height)
    else:
        self._left = int(self._left)
        self._top = int(self._top)
        self._width = int(self._width)
        self._height = int(self._height)