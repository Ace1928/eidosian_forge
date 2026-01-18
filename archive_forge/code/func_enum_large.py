important invariant is that the parts on the stack are themselves in
def enum_large(self, multiplicities, lb):
    """Enumerate the partitions of a multiset with lb < num(parts)

        Equivalent to enum_range(multiplicities, lb, sum(multiplicities))

        Parameters
        ==========

        multiplicities
            list of multiplicities of the components of the multiset.

        lb
            Number of parts in the partition must be greater than
            this lower bound.


        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_large([2,2], 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a'], ['b'], ['b']],
        [['a', 'b'], ['a'], ['b']],
        [['a'], ['a'], ['b', 'b']],
        [['a'], ['a'], ['b'], ['b']]]

        See Also
        ========

        enum_all, enum_small, enum_range

        """
    self.discarded = 0
    if lb >= sum(multiplicities):
        return
    self._initialize_enumeration(multiplicities)
    self.decrement_part_large(self.top_part(), 0, lb)
    while True:
        good_partition = True
        while self.spread_part_multiplicity():
            if not self.decrement_part_large(self.top_part(), 0, lb):
                self.discarded += 1
                good_partition = False
                break
        if good_partition:
            state = [self.f, self.lpart, self.pstack]
            yield state
        while not self.decrement_part_large(self.top_part(), 1, lb):
            if self.lpart == 0:
                return
            self.lpart -= 1