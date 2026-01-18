import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def candidate_index(self, node):
    """
        Find and return the promotion candidate and its index.

        Return (None, None) if no valid candidate was found.
        """
    index = node.first_child_not_matching_class(nodes.PreBibliographic)
    if index is None or len(node) > index + 1 or (not isinstance(node[index], nodes.section)):
        return (None, None)
    else:
        return (node[index], index)