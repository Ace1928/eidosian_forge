from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def sequence_relation(self, elem):
    """Create sequence relationship object, relationship between two sequences."""
    return PX.SequenceRelation(elem.get('type'), elem.get('id_ref_0'), elem.get('id_ref_1'), distance=_float(elem.get('distance')), confidence=_get_child_as(elem, 'confidence', self.confidence))