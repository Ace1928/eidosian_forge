import os
import sys
from mmap import mmap, ACCESS_READ
from mmap import ALLOCATIONGRANULARITY
def extend_right_to(self, window, max_size):
    """Adjust the size to make our window end where the right window begins, but don't
        get larger than max_size"""
    self.size = min(self.size + (window.ofs - self.ofs_end()), max_size)