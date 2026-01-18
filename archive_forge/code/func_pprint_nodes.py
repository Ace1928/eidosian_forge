def pprint_nodes(subtrees):
    """
    Prettyprints systems of nodes.

    Examples
    ========

    >>> from sympy.printing.tree import pprint_nodes
    >>> print(pprint_nodes(["a", "b1\\nb2", "c"]))
    +-a
    +-b1
    | b2
    +-c

    """

    def indent(s, type=1):
        x = s.split('\n')
        r = '+-%s\n' % x[0]
        for a in x[1:]:
            if a == '':
                continue
            if type == 1:
                r += '| %s\n' % a
            else:
                r += '  %s\n' % a
        return r
    if not subtrees:
        return ''
    f = ''
    for a in subtrees[:-1]:
        f += indent(a)
    f += indent(subtrees[-1], 2)
    return f