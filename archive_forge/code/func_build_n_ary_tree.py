from math import ceil, log
def build_n_ary_tree(leaves, n):
    """Build N-ary tree from sequence of leaf nodes.

    Return a list of lists where each non-leaf node is a list containing
    max n nodes.
    """
    if not leaves:
        return []
    assert n > 1
    depth = ceil(log(len(leaves), n))
    if depth <= 1:
        return list(leaves)
    root = []
    unassigned = None
    full_step = n ** (depth - 1)
    for i in range(0, len(leaves), full_step):
        subtree = leaves[i:i + full_step]
        if len(subtree) < full_step:
            unassigned = subtree
            break
        while len(subtree) > n:
            subtree = [subtree[k:k + n] for k in range(0, len(subtree), n)]
        root.append(subtree)
    if unassigned:
        subtree = build_n_ary_tree(unassigned, n)
        if len(subtree) <= n - len(root):
            root.extend(subtree)
        else:
            root.append(subtree)
        assert len(root) <= n
    return root