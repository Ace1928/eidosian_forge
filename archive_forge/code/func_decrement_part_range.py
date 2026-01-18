important invariant is that the parts on the stack are themselves in
def decrement_part_range(self, part, lb, ub):
    """Decrements part (a subrange of pstack), if possible, returning
        True iff the part was successfully decremented.

        Parameters
        ==========

         part
            part to be decremented (topmost part on the stack)

        ub
            the maximum number of parts allowed in a partition
            returned by the calling traversal.

        lb
            The partitions produced by the calling enumeration must
            have more parts than this value.

        Notes
        =====

        Combines the constraints of _small and _large decrement
        methods.  If returns success, part has been decremented at
        least once, but perhaps by quite a bit more if needed to meet
        the lb constraint.
        """
    return self.decrement_part_small(part, ub) and self.decrement_part_large(part, 0, lb)