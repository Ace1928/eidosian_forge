import re
import html
from paste.util import PySourceColor
def format_extra_data(self, importance, title, value):
    if isinstance(value, str):
        s = self.pretty_string_repr(value)
        if '\n' in s:
            return '%s:<br><pre>%s</pre>' % (title, self.quote(s))
        else:
            return '%s: <tt>%s</tt>' % (title, self.quote(s))
    elif isinstance(value, dict):
        return self.zebra_table(title, value)
    elif isinstance(value, (list, tuple)) and self.long_item_list(value):
        return '%s: <tt>[<br>\n&nbsp; &nbsp; %s]</tt>' % (title, ',<br>&nbsp; &nbsp; '.join(map(self.quote, map(repr, value))))
    else:
        return '%s: <tt>%s</tt>' % (title, self.quote(repr(value)))