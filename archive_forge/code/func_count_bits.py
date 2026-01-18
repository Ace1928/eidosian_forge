from __future__ import absolute_import, division, print_function
import sys
def count_bits(no):
    """
        Given an integer, compute the number of bits necessary to store its absolute value.
        """
    no = abs(no)
    count = 0
    while no > 0:
        no >>= 1
        count += 1
    return count