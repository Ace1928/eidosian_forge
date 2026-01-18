import os
import random
import time
from ._compat import long, binary_type
def random_32(self):
    return self.random_16() * 65536 + self.random_16()