import re
import docutils
from docutils import nodes, writers, languages
def new_row(self):
    self._rows.append([])