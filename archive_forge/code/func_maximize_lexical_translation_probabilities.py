from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def maximize_lexical_translation_probabilities(self, counts):
    for t, src_words in counts.t_given_s.items():
        for s in src_words:
            estimate = counts.t_given_s[t][s] / counts.any_t_given_s[s]
            self.translation_table[t][s] = max(estimate, IBMModel.MIN_PROB)