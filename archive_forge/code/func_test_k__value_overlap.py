import unittest
import pygame.constants
def test_k__value_overlap(self):
    """Ensures no unexpected K constant values overlap."""
    EXPECTED_OVERLAPS = {frozenset(('K_' + n for n in item)) for item in K_AND_KSCAN_COMMON_OVERLAPS}
    overlaps = create_overlap_set(self.K_NAMES)
    self.assertSetEqual(overlaps, EXPECTED_OVERLAPS)