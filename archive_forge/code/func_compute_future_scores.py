import warnings
from collections import defaultdict
from math import log
def compute_future_scores(self, src_sentence):
    """
        Determines the approximate scores for translating every
        subsequence in ``src_sentence``

        Future scores can be used a look-ahead to determine the
        difficulty of translating the remaining parts of a src_sentence.

        :type src_sentence: tuple(str)

        :return: Scores of subsequences referenced by their start and
            end positions. For example, result[2][5] is the score of the
            subsequence covering positions 2, 3, and 4.
        :rtype: dict(int: (dict(int): float))
        """
    scores = defaultdict(lambda: defaultdict(lambda: float('-inf')))
    for seq_length in range(1, len(src_sentence) + 1):
        for start in range(0, len(src_sentence) - seq_length + 1):
            end = start + seq_length
            phrase = src_sentence[start:end]
            if phrase in self.phrase_table:
                score = self.phrase_table.translations_for(phrase)[0].log_prob
                score += self.language_model.probability(phrase)
                scores[start][end] = score
            for mid in range(start + 1, end):
                combined_score = scores[start][mid] + scores[mid][end]
                if combined_score > scores[start][end]:
                    scores[start][end] = combined_score
    return scores