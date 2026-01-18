import re
import docutils
from docutils import nodes, writers, languages
def list_end(self):
    self.dedent()
    self._list_char.pop()