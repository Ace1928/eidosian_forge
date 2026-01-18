import warnings
from collections import defaultdict
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel1
from nltk.translate.ibm_model import Counts
def prob_all_alignments(self, src_sentence, trg_sentence):
    """
        Computes the probability of all possible word alignments,
        expressed as a marginal distribution over target words t

        Each entry in the return value represents the contribution to
        the total alignment probability by the target word t.

        To obtain probability(alignment | src_sentence, trg_sentence),
        simply sum the entries in the return value.

        :return: Probability of t for all s in ``src_sentence``
        :rtype: dict(str): float
        """
    alignment_prob_for_t = defaultdict(lambda: 0.0)
    for j in range(1, len(trg_sentence)):
        t = trg_sentence[j]
        for i in range(0, len(src_sentence)):
            alignment_prob_for_t[t] += self.prob_alignment_point(i, j, src_sentence, trg_sentence)
    return alignment_prob_for_t