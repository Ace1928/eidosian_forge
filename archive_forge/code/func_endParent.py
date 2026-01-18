import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def endParent(self):
    """End an XML element with children."""
    name = self._parent_stack.pop()
    self._level -= self._increment
    self.ignorableWhitespace(self._indent * self._level)
    self.endElement(name)