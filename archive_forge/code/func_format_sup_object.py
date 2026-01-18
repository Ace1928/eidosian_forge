import re
import html
from paste.util import PySourceColor
def format_sup_object(self, obj):
    return 'In object: %s' % self.emphasize(self.quote(repr(obj)))