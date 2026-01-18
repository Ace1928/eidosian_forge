import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def remove_component(self, value):
    """Remove the entry with the passed ID from the group."""
    self.components.remove(value)