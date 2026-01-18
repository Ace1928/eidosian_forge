import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def maximize_distortion_probabilities(self, counts):
    head_d_table = self.head_distortion_table
    for dj, src_classes in counts.head_distortion.items():
        for s_cls, trg_classes in src_classes.items():
            for t_cls in trg_classes:
                estimate = counts.head_distortion[dj][s_cls][t_cls] / counts.head_distortion_for_any_dj[s_cls][t_cls]
                head_d_table[dj][s_cls][t_cls] = max(estimate, IBMModel.MIN_PROB)
    non_head_d_table = self.non_head_distortion_table
    for dj, trg_classes in counts.non_head_distortion.items():
        for t_cls in trg_classes:
            estimate = counts.non_head_distortion[dj][t_cls] / counts.non_head_distortion_for_any_dj[t_cls]
            non_head_d_table[dj][t_cls] = max(estimate, IBMModel.MIN_PROB)