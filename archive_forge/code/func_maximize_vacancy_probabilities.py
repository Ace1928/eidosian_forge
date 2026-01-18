import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel4
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def maximize_vacancy_probabilities(self, counts):
    MIN_PROB = IBMModel.MIN_PROB
    head_vacancy_table = self.head_vacancy_table
    for dv, max_vs in counts.head_vacancy.items():
        for max_v, trg_classes in max_vs.items():
            for t_cls in trg_classes:
                estimate = counts.head_vacancy[dv][max_v][t_cls] / counts.head_vacancy_for_any_dv[max_v][t_cls]
                head_vacancy_table[dv][max_v][t_cls] = max(estimate, MIN_PROB)
    non_head_vacancy_table = self.non_head_vacancy_table
    for dv, max_vs in counts.non_head_vacancy.items():
        for max_v, trg_classes in max_vs.items():
            for t_cls in trg_classes:
                estimate = counts.non_head_vacancy[dv][max_v][t_cls] / counts.non_head_vacancy_for_any_dv[max_v][t_cls]
                non_head_vacancy_table[dv][max_v][t_cls] = max(estimate, MIN_PROB)