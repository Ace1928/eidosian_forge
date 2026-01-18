important invariant is that the parts on the stack are themselves in
def count_partitions_slow(self, multiplicities):
    """Returns the number of partitions of a multiset whose elements
        have the multiplicities given in ``multiplicities``.

        Primarily for comparison purposes.  It follows the same path as
        enumerate, and counts, rather than generates, the partitions.

        See Also
        ========

        count_partitions
            Has the same calling interface, but is much faster.

        """
    self.pcount = 0
    self._initialize_enumeration(multiplicities)
    while True:
        while self.spread_part_multiplicity():
            pass
        self.pcount += 1
        while not self.decrement_part(self.top_part()):
            if self.lpart == 0:
                return self.pcount
            self.lpart -= 1