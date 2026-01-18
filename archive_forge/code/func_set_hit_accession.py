from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def set_hit_accession(self):
    """Record the accession value of the database sequence (PRIVATE)."""
    self._hit.accession = self._value
    self._descr.accession = self._value