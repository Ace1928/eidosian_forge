from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def endDBRefElement(self, name, qname):
    """Handle the end of a DBRef element."""
    namespace, localname = name
    if namespace is not None:
        raise RuntimeError(f"Unexpected namespace '{namespace}' for DBRef element")
    if qname is not None:
        raise RuntimeError(f"Unexpected qname '{qname}' for DBRef element")
    if localname != 'DBRef':
        raise RuntimeError(f"Unexpected localname '{localname}' for DBRef element")
    if self.data:
        raise RuntimeError(f"Unexpected data received for DBRef element: '{self.data}'")
    self.data = None
    self.endElementNS = self.endEntryElement