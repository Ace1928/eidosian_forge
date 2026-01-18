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
def characterDataHandlerEscape(self, content):
    """Handle character data by encoding it."""
    content = escape(content)
    self.data.append(content)