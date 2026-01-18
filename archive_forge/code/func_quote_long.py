import re
import html
from paste.util import PySourceColor
def quote_long(self, s):
    return '<pre>%s</pre>' % self.quote(s)