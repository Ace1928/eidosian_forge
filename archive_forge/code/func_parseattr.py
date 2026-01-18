import re
def parseattr(self, word, rest):
    if word == 'FontBBox':
        l, b, r, t = [int(thing) for thing in rest.split()]
        self._attrs[word] = (l, b, r, t)
    elif word == 'Comment':
        self._comments.append(rest)
    else:
        try:
            value = int(rest)
        except (ValueError, OverflowError):
            self._attrs[word] = rest
        else:
            self._attrs[word] = value