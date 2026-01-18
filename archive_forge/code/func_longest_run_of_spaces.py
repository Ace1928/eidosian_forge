import os
import sys
import tokenize
def longest_run_of_spaces(self):
    count, trailing = self.norm
    return max(len(count) - 1, trailing)