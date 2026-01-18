from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def startSequenceElement(self, attrs):
    """Parse DNA, RNA, or protein sequence."""
    if attrs:
        raise ValueError('Unexpected attributes found in sequence element')
    if self.data is not None:
        raise RuntimeError(f"Unexpected data found: '{self.data}'")
    self.data = ''
    self.endElementNS = self.endSequenceElement