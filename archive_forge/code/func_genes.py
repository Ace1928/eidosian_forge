import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
@property
def genes(self):
    """Get a list of entries of type gene."""
    return [e for e in self.entries.values() if e.type == 'gene']