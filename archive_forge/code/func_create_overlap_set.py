import unittest
import pygame.constants
def create_overlap_set(constant_names):
    """Helper function to find overlapping constant values/names.

    Returns a set of fronzensets:
        set(frozenset(names of overlapping constants), ...)
    """
    overlap_dict = {}
    for name in constant_names:
        value = getattr(pygame.constants, name)
        overlap_dict.setdefault(value, set()).add(name)
    overlaps = set()
    for overlap_names in overlap_dict.values():
        if len(overlap_names) > 1:
            overlaps.add(frozenset(overlap_names))
    return overlaps