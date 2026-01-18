from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def set_hit_len(self):
    """Record the length of the hit."""
    self._hit.length = int(self._value)