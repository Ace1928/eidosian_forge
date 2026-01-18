import re
import itertools
def dataBitIterator(self, data):
    if not self._dataBitList:
        self._dataBitList = list(self._dataBitIterator(data))
    return iter(self._dataBitList)