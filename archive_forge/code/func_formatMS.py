import re, time, datetime
from .utils import isStr
def formatMS(self, fmt):
    """format like MS date using the notation
        {YY}    --> 2 digit year
        {YYYY}  --> 4 digit year
        {M}     --> month as digit
        {MM}    --> 2 digit month
        {MMM}   --> abbreviated month name
        {MMMM}  --> monthname
        {MMMMM} --> first character of monthname
        {D}     --> day of month as digit
        {DD}    --> 2 digit day of month
        {DDD}   --> abrreviated weekday name
        {DDDD}  --> weekday name
        """
    r = fmt[:]
    f = 0
    while 1:
        m = _fmtPat.search(r, f)
        if m:
            y = getattr(self, '_fmt' + m.group()[1:-1].upper())()
            i, j = m.span()
            r = r[0:i] + y + r[j:]
            f = i + len(y)
        else:
            return r