import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
@property
def reactant_ids(self):
    """Return a list of substrate and product reactant IDs."""
    return self._products.union(self._substrates)