def merge_regions(self):
    """Return sequences of matching and conflicting regions.

        This returns tuples, where the first value says what kind we
        have:

        'unchanged', start, end
             Take a region of base[start:end]

        'same', astart, aend
             b and a are different from base but give the same result

        'a', start, end
             Non-clashing insertion from a[start:end]

        Method is as follows:

        The two sequences align only on regions which match the base
        and both descendents.  These are found by doing a two-way diff
        of each one against the base, and then finding the
        intersections between those regions.  These "sync regions"
        are by definition unchanged in both and easily dealt with.

        The regions in between can be in any of three cases:
        conflicted, or changed on only one side.
        """
    iz = ia = ib = 0
    for zmatch, zend, amatch, aend, bmatch, bend in self.find_sync_regions():
        matchlen = zend - zmatch
        len_a = amatch - ia
        len_b = bmatch - ib
        if len_a or len_b:
            same = compare_range(self.a, ia, amatch, self.b, ib, bmatch)
            if same:
                yield ('same', ia, amatch)
            else:
                equal_a = compare_range(self.a, ia, amatch, self.base, iz, zmatch)
                equal_b = compare_range(self.b, ib, bmatch, self.base, iz, zmatch)
                if equal_a and (not equal_b):
                    yield ('b', ib, bmatch)
                elif equal_b and (not equal_a):
                    yield ('a', ia, amatch)
                elif not equal_a and (not equal_b):
                    if self.is_cherrypick:
                        for node in self._refine_cherrypick_conflict(iz, zmatch, ia, amatch, ib, bmatch):
                            yield node
                    else:
                        yield ('conflict', iz, zmatch, ia, amatch, ib, bmatch)
                else:
                    raise AssertionError("can't handle a=b=base but unmatched")
            ia = amatch
            ib = bmatch
        iz = zmatch
        if matchlen > 0:
            yield ('unchanged', zmatch, zend)
            iz = zend
            ia = aend
            ib = bend