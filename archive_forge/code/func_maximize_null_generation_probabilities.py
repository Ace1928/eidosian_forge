from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def maximize_null_generation_probabilities(self, counts):
    p1_estimate = counts.p1 / (counts.p1 + counts.p0)
    p1_estimate = max(p1_estimate, IBMModel.MIN_PROB)
    self.p1 = min(p1_estimate, 1 - IBMModel.MIN_PROB)