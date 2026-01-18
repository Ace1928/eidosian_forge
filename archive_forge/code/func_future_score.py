import warnings
from collections import defaultdict
from math import log
def future_score(self, hypothesis, future_score_table, sentence_length):
    """
        Determines the approximate score for translating the
        untranslated words in ``hypothesis``
        """
    score = 0.0
    for span in hypothesis.untranslated_spans(sentence_length):
        score += future_score_table[span[0]][span[1]]
    return score