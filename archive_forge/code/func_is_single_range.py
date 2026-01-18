def is_single_range(self):
    """Does this range specifier consist of only a single range set?"""
    return len(self.range_specs) == 1