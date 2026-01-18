import re
import html
from paste.util import PySourceColor
def format_sup_expression(self, expr):
    return self.emphasize('In expression: %s' % self.quote(expr))