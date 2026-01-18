from pythran.analyses import PotentialIterator, Aliases
from pythran.passmanager import Transformation
from pythran.utils import path_to_attr, path_to_node
def find_matching_builtin(self, node):
    """
        Return matched keyword.

        If the node alias on a correct keyword (and only it), it matches.
        """
    for path in EQUIVALENT_ITERATORS.keys():
        correct_alias = {path_to_node(path)}
        if self.aliases[node.func] == correct_alias:
            return path