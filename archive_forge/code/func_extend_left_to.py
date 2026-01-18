import os
import sys
from mmap import mmap, ACCESS_READ
from mmap import ALLOCATIONGRANULARITY
def extend_left_to(self, window, max_size):
    """Adjust the offset to start where the given window on our left ends if possible,
        but don't make yourself larger than max_size.
        The resize will assure that the new window still contains the old window area"""
    rofs = self.ofs - window.ofs_end()
    nsize = rofs + self.size
    rofs -= nsize - min(nsize, max_size)
    self.ofs -= rofs
    self.size += rofs