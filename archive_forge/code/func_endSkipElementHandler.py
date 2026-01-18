import os
import warnings
from collections import Counter
from xml.parsers import expat
from io import BytesIO
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from urllib.request import urlopen
from urllib.parse import urlparse
from Bio import StreamModeError
def endSkipElementHandler(self, name):
    """Handle end of an XML skip element."""
    self.level -= 1
    if self.level == 0:
        self.parser.StartElementHandler = self.startElementHandler
        self.parser.EndElementHandler = self.endElementHandler