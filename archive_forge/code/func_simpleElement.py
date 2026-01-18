import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def simpleElement(self, name, content=None):
    """Create an XML element without children with the given content."""
    self.startElement(name, attrs={})
    if content:
        self.characters(content)
    self.endElement(name)