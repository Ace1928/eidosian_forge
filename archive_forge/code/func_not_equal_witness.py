import os
import sys
import tokenize
def not_equal_witness(self, other):
    n = max(self.longest_run_of_spaces(), other.longest_run_of_spaces()) + 1
    a = []
    for ts in range(1, n + 1):
        if self.indent_level(ts) != other.indent_level(ts):
            a.append((ts, self.indent_level(ts), other.indent_level(ts)))
    return a