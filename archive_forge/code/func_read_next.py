import re
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
def read_next(self, rstrip=True):
    """Return the next non-empty line, trailing whitespace removed."""
    if len(self.buf) > 0:
        return self.buf.pop()
    self.line = self.handle.readline()
    while self.line and rstrip and (not self.line.strip()):
        self.line = self.handle.readline()
    if self.line:
        if rstrip:
            self.line = self.line.rstrip()
    return self.line