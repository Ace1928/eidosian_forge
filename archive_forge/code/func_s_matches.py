import re
from IPython.core.hooks import CommandChainDispatcher
def s_matches(self, key):
    if key not in self.strs:
        return
    for el in self.strs[key]:
        yield el[1]