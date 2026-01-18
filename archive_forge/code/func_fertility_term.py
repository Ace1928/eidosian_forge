import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def fertility_term():
    value = 1.0
    src_sentence = alignment_info.src_sentence
    for i in range(1, len(src_sentence)):
        fertility = alignment_info.fertility_of_i(i)
        value *= factorial(fertility) * ibm_model.fertility_table[fertility][src_sentence[i]]
        if value < MIN_PROB:
            return MIN_PROB
    return value