from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def modified_scan_and_fill(self, alpha, w, y):
    self.modified_scan(alpha, w, y, fill=True)