import re
from nltk.grammar import Nonterminal, Production
from nltk.internals import deprecated
def leaf_treeposition(self, index):
    """
        :return: The tree position of the ``index``-th leaf in this
            tree.  I.e., if ``tp=self.leaf_treeposition(i)``, then
            ``self[tp]==self.leaves()[i]``.

        :raise IndexError: If this tree contains fewer than ``index+1``
            leaves, or if ``index<0``.
        """
    if index < 0:
        raise IndexError('index must be non-negative')
    stack = [(self, ())]
    while stack:
        value, treepos = stack.pop()
        if not isinstance(value, Tree):
            if index == 0:
                return treepos
            else:
                index -= 1
        else:
            for i in range(len(value) - 1, -1, -1):
                stack.append((value[i], treepos + (i,)))
    raise IndexError('index must be less than or equal to len(self)')