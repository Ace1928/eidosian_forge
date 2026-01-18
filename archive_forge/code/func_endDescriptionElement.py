from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def endDescriptionElement(self, name, qname):
    """Handle the end of a description element."""
    namespace, localname = name
    if namespace is not None:
        raise RuntimeError(f"Unexpected namespace '{namespace}' for description end")
    if qname is not None:
        raise RuntimeError(f"Unexpected qname '{qname}' for description end")
    if localname != 'description':
        raise RuntimeError('Failed to find end of description element')
    record = self.records[-1]
    description = self.data
    if description:
        record.description = description
    self.data = None
    self.endElementNS = self.endEntryElement