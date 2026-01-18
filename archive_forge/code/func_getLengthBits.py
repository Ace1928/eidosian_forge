import re
import itertools
def getLengthBits(self, ver):
    if 0 < ver < 10:
        return self.lengthbits[0]
    elif ver < 27:
        return self.lengthbits[1]
    elif ver < 41:
        return self.lengthbits[2]
    raise ValueError('Unknown version: ' + ver)