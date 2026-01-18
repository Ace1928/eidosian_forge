import time
from . import debug, errors, osutils, revision, trace
def find_unique_lca(self, left_revision, right_revision, count_steps=False):
    """Find a unique LCA.

        Find lowest common ancestors.  If there is no unique  common
        ancestor, find the lowest common ancestors of those ancestors.

        Iteration stops when a unique lowest common ancestor is found.
        The graph origin is necessarily a unique lowest common ancestor.

        Note that None is not an acceptable substitute for NULL_REVISION.
        in the input for this method.

        :param count_steps: If True, the return value will be a tuple of
            (unique_lca, steps) where steps is the number of times that
            find_lca was run.  If False, only unique_lca is returned.
        """
    revisions = [left_revision, right_revision]
    steps = 0
    while True:
        steps += 1
        lca = self.find_lca(*revisions)
        if len(lca) == 1:
            result = lca.pop()
            if count_steps:
                return (result, steps)
            else:
                return result
        if len(lca) == 0:
            raise errors.NoCommonAncestor(left_revision, right_revision)
        revisions = lca