from statsmodels.compat.python import lrange
import numpy as np
def branch2(tree):
    """walking a tree bottom-up based on dictionary
    """
    if isinstance(tree, tuple):
        name, subtree = tree
        print(name, data2[name])
        print('subtree', subtree)
        if testxb:
            branchsum = data2[name]
        else:
            branchsum = name
        for b in subtree:
            branchsum = branchsum + branch2(b)
    else:
        leavessum = sum((data2[bi] for bi in tree))
        print('final branch with', tree, ''.join(tree), leavessum)
        if testxb:
            return leavessum
        else:
            return ''.join(tree)
    print('working on branch', tree, branchsum)
    return branchsum