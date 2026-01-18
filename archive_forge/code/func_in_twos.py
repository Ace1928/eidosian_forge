from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def in_twos(L):
    assert len(L) % 2 == 0
    return [L[i:i + 2] for i in range(0, len(L), 2)]