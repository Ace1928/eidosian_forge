from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def startPropertyElement(self, attrs):
    """Handle the start of a property element."""
    property_name = None
    property_value = None
    for key, value in attrs.items():
        namespace, localname = key
        if namespace is None:
            if localname == 'name':
                property_name = value
            elif localname == 'value':
                property_value = value
            else:
                raise ValueError("Unexpected attribute '%s' found for property element", key)
        else:
            raise ValueError(f"Unexpected namespace '{namespace}' for property attribute")
    if property_name is None:
        raise ValueError('Failed to find name for property element')
    record = self.records[-1]
    if property_name == 'molecule_type':
        assert record.annotations[property_name] in property_value
        record.annotations[property_name] = property_value
    else:
        if property_name not in record.annotations:
            record.annotations[property_name] = []
        record.annotations[property_name].append(property_value)
    self.endElementNS = self.endPropertyElement