import re
def isMatch(self, labels):
    if self._operator == 'in':
        return self._key in labels and labels.get(self._key) in self._data
    elif self._operator == 'notin':
        return self._key not in labels or labels.get(self._key) not in self._data
    else:
        return self._key not in labels if self._operator == '!' else self._key in labels