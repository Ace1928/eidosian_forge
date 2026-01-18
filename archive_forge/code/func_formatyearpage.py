import sys
import datetime
import locale as _locale
from itertools import repeat
def formatyearpage(self, theyear, width=3, css='calendar.css', encoding=None):
    """
        Return a formatted year as a complete HTML page.
        """
    if encoding is None:
        encoding = sys.getdefaultencoding()
    v = []
    a = v.append
    a('<?xml version="1.0" encoding="%s"?>\n' % encoding)
    a('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n')
    a('<html>\n')
    a('<head>\n')
    a('<meta http-equiv="Content-Type" content="text/html; charset=%s" />\n' % encoding)
    if css is not None:
        a('<link rel="stylesheet" type="text/css" href="%s" />\n' % css)
    a('<title>Calendar for %d</title>\n' % theyear)
    a('</head>\n')
    a('<body>\n')
    a(self.formatyear(theyear, width))
    a('</body>\n')
    a('</html>\n')
    return ''.join(v).encode(encoding, 'xmlcharrefreplace')