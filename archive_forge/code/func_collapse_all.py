import collections
import copy
import itertools
import random
import re
import warnings
def collapse_all(self, target=None, **kwargs):
    """Collapse all the descendents of this tree, leaving only terminals.

        Total branch lengths are preserved, i.e. the distance to each terminal
        stays the same.

        For example, this will safely collapse nodes with poor bootstrap
        support:

            >>> from Bio import Phylo
            >>> tree = Phylo.read('PhyloXML/apaf.xml', 'phyloxml')
            >>> print("Total branch length %0.2f" % tree.total_branch_length())
            Total branch length 20.44
            >>> tree.collapse_all(lambda c: c.confidence is not None and c.confidence < 70)
            >>> print("Total branch length %0.2f" % tree.total_branch_length())
            Total branch length 21.37

        This implementation avoids strange side-effects by using level-order
        traversal and testing all clade properties (versus the target
        specification) up front. In particular, if a clade meets the target
        specification in the original tree, it will be collapsed.  For example,
        if the condition is:

            >>> from Bio import Phylo
            >>> tree = Phylo.read('PhyloXML/apaf.xml', 'phyloxml')
            >>> print("Total branch length %0.2f" % tree.total_branch_length())
            Total branch length 20.44
            >>> tree.collapse_all(lambda c: c.branch_length < 0.1)
            >>> print("Total branch length %0.2f" % tree.total_branch_length())
            Total branch length 21.13

        Collapsing a clade's parent node adds the parent's branch length to the
        child, so during the execution of collapse_all, a clade's branch_length
        may increase. In this implementation, clades are collapsed according to
        their properties in the original tree, not the properties when tree
        traversal reaches the clade. (It's easier to debug.) If you want the
        other behavior (incremental testing), modifying the source code of this
        function is straightforward.
        """
    matches = list(self.find_clades(target, False, 'level', **kwargs))
    if not matches:
        return
    if matches[0] == self.root:
        matches.pop(0)
    for clade in matches:
        self.collapse(clade)