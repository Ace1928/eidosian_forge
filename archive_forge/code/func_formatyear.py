import sys
import datetime
import locale as _locale
from itertools import repeat
def formatyear(self, theyear, width=3):
    """
        Return a formatted year as a table of tables.
        """
    v = []
    a = v.append
    width = max(width, 1)
    a('<table border="0" cellpadding="0" cellspacing="0" class="%s">' % self.cssclass_year)
    a('\n')
    a('<tr><th colspan="%d" class="%s">%s</th></tr>' % (width, self.cssclass_year_head, theyear))
    for i in range(January, January + 12, width):
        months = range(i, min(i + width, 13))
        a('<tr>')
        for m in months:
            a('<td>')
            a(self.formatmonth(theyear, m, withyear=False))
            a('</td>')
        a('</tr>')
    a('</table>')
    return ''.join(v)