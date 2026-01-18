from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def set_hit_def(self):
    """Record the definition line of the database sequence (PRIVATE)."""
    self._hit.hit_def = self._value
    self._hit.title += self._value
    self._descr.title = self._hit.title