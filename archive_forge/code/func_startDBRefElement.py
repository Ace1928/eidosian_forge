from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def startDBRefElement(self, attrs):
    """Parse a database cross reference."""
    source = None
    ID = None
    for key, value in attrs.items():
        namespace, localname = key
        if namespace is None:
            if localname == 'source':
                source = value
            elif localname == 'id':
                ID = value
            else:
                raise ValueError(f"Unexpected attribute '{key}' found for DBRef element")
        else:
            raise ValueError(f"Unexpected namespace '{namespace}' for DBRef attribute")
    if source is None:
        raise ValueError('Failed to find source for DBRef element')
    if ID is None:
        raise ValueError('Failed to find id for DBRef element')
    if self.data is not None:
        raise RuntimeError(f"Unexpected data found: '{self.data}'")
    self.data = ''
    record = self.records[-1]
    dbxref = f'{source}:{ID}'
    if dbxref not in record.dbxrefs:
        record.dbxrefs.append(dbxref)
    self.endElementNS = self.endDBRefElement