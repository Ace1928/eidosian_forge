import re
import html
from paste.util import PySourceColor
def format_sup_info(self, info):
    return [self.quote_long(info)]