from __future__ import absolute_import, division, print_function
import sys
def simple_gcd(a, b):
    """Compute GCD of its two inputs."""
    while b != 0:
        a, b = (b, a % b)
    return a