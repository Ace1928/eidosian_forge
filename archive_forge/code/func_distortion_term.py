import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def distortion_term(j):
    t = alignment_info.trg_sentence[j]
    i = alignment_info.alignment[j]
    if i == 0:
        return 1.0
    if alignment_info.is_head_word(j):
        previous_cept = alignment_info.previous_cept(j)
        src_class = None
        if previous_cept is not None:
            previous_s = alignment_info.src_sentence[previous_cept]
            src_class = ibm_model.src_classes[previous_s]
        trg_class = ibm_model.trg_classes[t]
        dj = j - alignment_info.center_of_cept(previous_cept)
        return ibm_model.head_distortion_table[dj][src_class][trg_class]
    previous_position = alignment_info.previous_in_tablet(j)
    trg_class = ibm_model.trg_classes[t]
    dj = j - previous_position
    return ibm_model.non_head_distortion_table[dj][trg_class]