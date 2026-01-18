from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def bc_getter(elem):
    """Get binary characters from subnodes."""
    return _get_children_text(elem, 'bc')