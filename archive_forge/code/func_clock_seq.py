import os
import sys
from enum import Enum, _simple_enum
@property
def clock_seq(self):
    return (self.clock_seq_hi_variant & 63) << 8 | self.clock_seq_low