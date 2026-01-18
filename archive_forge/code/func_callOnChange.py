import doctest
import collections
def callOnChange(self, oldLeft, oldTop, oldWidth, oldHeight):
    if self.onChange is not None:
        self.onChange(Box(oldLeft, oldTop, oldWidth, oldHeight), Box(self._left, self._top, self._width, self._height))