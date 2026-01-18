def is_contiguous(self):
    """Can the collection of range_specs be coalesced into a single contiguous range?"""
    if len(self.range_specs) <= 1:
        return True
    merged = self.range_specs[0].copy()
    for s in self.range_specs[1:]:
        try:
            merged.merge_with(s)
        except:
            return False
    return True