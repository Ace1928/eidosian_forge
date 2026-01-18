import os
import sys
from mmap import mmap, ACCESS_READ
from mmap import ALLOCATIONGRANULARITY
@classmethod
def from_region(cls, region):
    """:return: new window from a region"""
    return cls(region._b, region.size())