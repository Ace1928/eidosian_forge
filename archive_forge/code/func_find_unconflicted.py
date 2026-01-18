def find_unconflicted(self):
    """Return a list of ranges in base that are not conflicted."""
    am = self.sequence_matcher(None, self.base, self.a).get_matching_blocks()
    bm = self.sequence_matcher(None, self.base, self.b).get_matching_blocks()
    unc = []
    while am and bm:
        a1 = am[0][0]
        a2 = a1 + am[0][2]
        b1 = bm[0][0]
        b2 = b1 + bm[0][2]
        i = intersect((a1, a2), (b1, b2))
        if i:
            unc.append(i)
        if a2 < b2:
            del am[0]
        else:
            del bm[0]
    return unc