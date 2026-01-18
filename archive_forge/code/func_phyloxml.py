from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def phyloxml(self, obj):
    """Convert phyloxml to Etree element."""
    elem = ElementTree.Element('phyloxml', obj.attributes)
    for tree in obj.phylogenies:
        elem.append(self.phylogeny(tree))
    for otr in obj.other:
        elem.append(self.other(otr))
    return elem