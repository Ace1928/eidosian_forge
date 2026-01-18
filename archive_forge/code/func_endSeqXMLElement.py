from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def endSeqXMLElement(self, name, qname):
    """Handle end of the seqXML element."""
    namespace, localname = name
    if namespace is not None:
        raise RuntimeError(f"Unexpected namespace '{namespace}' for seqXML end")
    if qname is not None:
        raise RuntimeError(f"Unexpected qname '{qname}' for seqXML end")
    if localname != 'seqXML':
        raise RuntimeError('Failed to find end of seqXML element')
    self.startElementNS = None
    self.endElementNS = None