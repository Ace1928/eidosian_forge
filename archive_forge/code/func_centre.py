import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
@property
def centre(self):
    """Return the centre of the Graphics object as an (x, y) tuple."""
    return (0.5 * (self.bounds[0][0] + self.bounds[1][0]), 0.5 * (self.bounds[0][1] + self.bounds[1][1]))