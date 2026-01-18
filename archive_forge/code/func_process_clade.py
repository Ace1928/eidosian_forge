import re
from io import StringIO
from Bio.Phylo import Newick
def process_clade(self, clade):
    """Remove node's parent and return it. Final processing of parsed clade."""
    if clade.name and (not (self.values_are_confidence or self.comments_are_confidence)) and (clade.confidence is None) and clade.clades:
        clade.confidence = _parse_confidence(clade.name)
        if clade.confidence is not None:
            clade.name = None
    try:
        parent = clade.parent
    except AttributeError:
        pass
    else:
        parent.clades.append(clade)
        del clade.parent
        return parent