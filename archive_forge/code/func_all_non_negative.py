from collections import defaultdict
from itertools import chain
def all_non_negative(self):
    for v in self.values():
        if v < v * 0:
            return False
    return True