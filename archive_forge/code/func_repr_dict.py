import builtins
from itertools import islice
from _thread import get_ident
def repr_dict(self, x, level):
    n = len(x)
    if n == 0:
        return '{}'
    if level <= 0:
        return '{' + self.fillvalue + '}'
    newlevel = level - 1
    repr1 = self.repr1
    pieces = []
    for key in islice(_possibly_sorted(x), self.maxdict):
        keyrepr = repr1(key, newlevel)
        valrepr = repr1(x[key], newlevel)
        pieces.append('%s: %s' % (keyrepr, valrepr))
    if n > self.maxdict:
        pieces.append(self.fillvalue)
    s = ', '.join(pieces)
    return '{%s}' % (s,)