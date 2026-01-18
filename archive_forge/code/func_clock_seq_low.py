import os
import sys
from enum import Enum, _simple_enum
@property
def clock_seq_low(self):
    return self.int >> 48 & 255