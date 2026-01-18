import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
@property
def substrates(self):
    """Return list of substrate Entry elements."""
    return [self._pathway.entries[sid] for sid in self._substrates]