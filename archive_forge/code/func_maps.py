import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
@property
def maps(self):
    """Get a list of entries of type map."""
    return [e for e in self.entries.values() if e.type == 'map']