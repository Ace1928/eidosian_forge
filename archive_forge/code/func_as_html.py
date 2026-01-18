from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def as_html(self, **fmt_dict):
    """Return string.
        This is the default formatter for HTML tables.
        An HTML table formatter must accept as arguments
        a table and a format dictionary.
        """
    fmt = self._get_fmt('html', **fmt_dict)
    formatted_rows = ['<table class="simpletable">']
    if self.title:
        title = '<caption>%s</caption>' % self.title
        formatted_rows.append(title)
    formatted_rows.extend((row.as_string('html', **fmt) for row in self))
    formatted_rows.append('</table>')
    return '\n'.join(formatted_rows)