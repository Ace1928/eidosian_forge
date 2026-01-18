from io import StringIO
from antlr4.Token import Token
def removeRange(self, v):
    if v.start == v.stop - 1:
        self.removeOne(v.start)
    elif self.intervals is not None:
        k = 0
        for i in self.intervals:
            if v.stop <= i.start:
                return
            elif v.start > i.start and v.stop < i.stop:
                self.intervals[k] = range(i.start, v.start)
                x = range(v.stop, i.stop)
                self.intervals.insert(k, x)
                return
            elif v.start <= i.start and v.stop >= i.stop:
                self.intervals.pop(k)
                k -= 1
            elif v.start < i.stop:
                self.intervals[k] = range(i.start, v.start)
            elif v.stop < i.stop:
                self.intervals[k] = range(v.stop, i.stop)
            k += 1