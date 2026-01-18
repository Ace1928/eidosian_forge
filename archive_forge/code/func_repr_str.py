import builtins
from itertools import islice
from _thread import get_ident
def repr_str(self, x, level):
    s = builtins.repr(x[:self.maxstring])
    if len(s) > self.maxstring:
        i = max(0, (self.maxstring - 3) // 2)
        j = max(0, self.maxstring - 3 - i)
        s = builtins.repr(x[:i] + x[len(x) - j:])
        s = s[:i] + self.fillvalue + s[len(s) - j:]
    return s