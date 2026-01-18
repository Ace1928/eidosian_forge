from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def confidence(self, elem):
    """Create confidence object."""
    return PX.Confidence(_float(elem.text), elem.get('type'))