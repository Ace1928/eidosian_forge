important invariant is that the parts on the stack are themselves in
def enum_small(self, multiplicities, ub):
    """Enumerate multiset partitions with no more than ``ub`` parts.

        Equivalent to enum_range(multiplicities, 0, ub)

        Parameters
        ==========

        multiplicities
             list of multiplicities of the components of the multiset.

        ub
            Maximum number of parts

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_small([2,2], 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b', 'b']],
        [['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']]]

        The implementation is based, in part, on the answer given to
        exercise 69, in Knuth [AOCP]_.

        See Also
        ========

        enum_all, enum_large, enum_range

        """
    self.discarded = 0
    if ub <= 0:
        return
    self._initialize_enumeration(multiplicities)
    while True:
        while self.spread_part_multiplicity():
            self.db_trace('spread 1')
            if self.lpart >= ub:
                self.discarded += 1
                self.db_trace('  Discarding')
                self.lpart = ub - 2
                break
        else:
            state = [self.f, self.lpart, self.pstack]
            yield state
        while not self.decrement_part_small(self.top_part(), ub):
            self.db_trace('Failed decrement, going to backtrack')
            if self.lpart == 0:
                return
            self.lpart -= 1
            self.db_trace('Backtracked to')
        self.db_trace('decrement ok, about to expand')