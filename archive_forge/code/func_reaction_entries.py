import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
@property
def reaction_entries(self):
    """List of entries corresponding to each reaction in the pathway."""
    return [self.entries[i] for i in self._reactions]