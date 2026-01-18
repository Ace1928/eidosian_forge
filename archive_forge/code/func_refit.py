from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def refit(self):
    """Refit (normalize) the prefixes in the node."""
    self.refitNodes()
    self.refitMappings()