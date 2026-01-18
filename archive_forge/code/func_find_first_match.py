import collections
import io
import itertools
import os
from taskflow.types import graph
from taskflow.utils import iter_utils
from taskflow.utils import misc
def find_first_match(self, matcher, only_direct=False, include_self=True):
    """Finds the *first* node that matching callback returns true.

        This will search not only this node but also any children nodes (in
        depth first order, from right to left) and finally if nothing is
        matched then ``None`` is returned instead of a node object.

        :param matcher: callback that takes one positional argument (a node)
                        and returns true if it matches desired node or false
                        if not.
        :param only_direct: only look at current node and its
                            direct children (implies that this does not
                            search using depth first).
        :param include_self: include the current node during searching.

        :returns: the node that matched (or ``None``)
        """
    if only_direct:
        if include_self:
            it = itertools.chain([self], self.reverse_iter())
        else:
            it = self.reverse_iter()
    else:
        it = self.dfs_iter(include_self=include_self)
    return iter_utils.find_first_match(it, matcher)