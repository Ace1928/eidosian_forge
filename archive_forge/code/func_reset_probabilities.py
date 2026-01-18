import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def reset_probabilities(self):
    super().reset_probabilities()
    self.head_distortion_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: self.MIN_PROB)))
    '\n        dict[int][int][int]: float. Probability(displacement of head\n        word | word class of previous cept,target word class).\n        Values accessed as ``distortion_table[dj][src_class][trg_class]``.\n        '
    self.non_head_distortion_table = defaultdict(lambda: defaultdict(lambda: self.MIN_PROB))
    '\n        dict[int][int]: float. Probability(displacement of non-head\n        word | target word class).\n        Values accessed as ``distortion_table[dj][trg_class]``.\n        '