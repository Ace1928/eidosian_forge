important invariant is that the parts on the stack are themselves in
def decrement_part_small(self, part, ub):
    """Decrements part (a subrange of pstack), if possible, returning
        True iff the part was successfully decremented.

        Parameters
        ==========

        part
            part to be decremented (topmost part on the stack)

        ub
            the maximum number of parts allowed in a partition
            returned by the calling traversal.

        Notes
        =====

        The goal of this modification of the ordinary decrement method
        is to fail (meaning that the subtree rooted at this part is to
        be skipped) when it can be proved that this part can only have
        child partitions which are larger than allowed by ``ub``. If a
        decision is made to fail, it must be accurate, otherwise the
        enumeration will miss some partitions.  But, it is OK not to
        capture all the possible failures -- if a part is passed that
        should not be, the resulting too-large partitions are filtered
        by the enumeration one level up.  However, as is usual in
        constrained enumerations, failing early is advantageous.

        The tests used by this method catch the most common cases,
        although this implementation is by no means the last word on
        this problem.  The tests include:

        1) ``lpart`` must be less than ``ub`` by at least 2.  This is because
           once a part has been decremented, the partition
           will gain at least one child in the spread step.

        2) If the leading component of the part is about to be
           decremented, check for how many parts will be added in
           order to use up the unallocated multiplicity in that
           leading component, and fail if this number is greater than
           allowed by ``ub``.  (See code for the exact expression.)  This
           test is given in the answer to Knuth's problem 7.2.1.5.69.

        3) If there is *exactly* enough room to expand the leading
           component by the above test, check the next component (if
           it exists) once decrementing has finished.  If this has
           ``v == 0``, this next component will push the expansion over the
           limit by 1, so fail.
        """
    if self.lpart >= ub - 1:
        self.p1 += 1
        return False
    plen = len(part)
    for j in range(plen - 1, -1, -1):
        if j == 0 and (part[0].v - 1) * (ub - self.lpart) < part[0].u:
            self.k1 += 1
            return False
        if j == 0 and part[j].v > 1 or (j > 0 and part[j].v > 0):
            part[j].v -= 1
            for k in range(j + 1, plen):
                part[k].v = part[k].u
            if plen > 1 and part[1].v == 0 and (part[0].u - part[0].v == (ub - self.lpart - 1) * part[0].v):
                self.k2 += 1
                self.db_trace('Decrement fails test 3')
                return False
            return True
    return False