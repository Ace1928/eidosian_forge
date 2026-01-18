import re
from mako import exceptions
def start_source(self, lineno):
    if self.lineno not in self.source_map:
        self.source_map[self.lineno] = lineno