from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
def advance_next_nonspace(self):
    self.offset = self.next_nonspace
    self.column = self.next_nonspace_column
    self.partially_consumed_tab = False