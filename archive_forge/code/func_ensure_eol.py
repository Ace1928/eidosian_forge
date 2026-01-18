import re
import docutils
from docutils import nodes, writers, languages
def ensure_eol(self):
    """Ensure the last line in body is terminated by new line."""
    if len(self.body) > 0 and self.body[-1][-1] != '\n':
        self.body.append('\n')