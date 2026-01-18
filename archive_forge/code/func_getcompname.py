from collections import namedtuple
import warnings
def getcompname(self):
    if self._comptype == 'ULAW':
        return 'CCITT G.711 u-law'
    elif self._comptype == 'ALAW':
        return 'CCITT G.711 A-law'
    else:
        return 'not compressed'