import os
import random
import time
from ._compat import long, binary_type
def stir(self, entropy, already_locked=False):
    if not already_locked:
        self.lock.acquire()
    try:
        for c in entropy:
            if self.pool_index == self.hash_len:
                self.pool_index = 0
            b = c & 255
            self.pool[self.pool_index] ^= b
            self.pool_index += 1
    finally:
        if not already_locked:
            self.lock.release()