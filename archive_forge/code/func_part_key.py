important invariant is that the parts on the stack are themselves in
def part_key(part):
    """Helper for MultisetPartitionTraverser.count_partitions that
    creates a key for ``part``, that only includes information which can
    affect the count for that part.  (Any irrelevant information just
    reduces the effectiveness of dynamic programming.)

    Notes
    =====

    This member function is a candidate for future exploration. There
    are likely symmetries that can be exploited to coalesce some
    ``part_key`` values, and thereby save space and improve
    performance.

    """
    rval = []
    for ps in part:
        rval.append(ps.u)
        rval.append(ps.v)
    return tuple(rval)