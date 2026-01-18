from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def binary_characters(self, obj):
    """Serialize a binary_characters node and its subnodes."""
    elem = ElementTree.Element('binary_characters', _clean_attrib(obj, ('type', 'gained_count', 'lost_count', 'present_count', 'absent_count')))
    for subn in ('gained', 'lost', 'present', 'absent'):
        subelem = ElementTree.Element(subn)
        for token in getattr(obj, subn):
            subelem.append(self.bc(token))
        elem.append(subelem)
    return elem