import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
def evaluate_word_pairs(self, pairs, delimiter='\t', encoding='utf8', restrict_vocab=300000, case_insensitive=True, dummy4unknown=False):
    """Compute correlation of the model with human similarity judgments.

        Notes
        -----
        More datasets can be found at
        * http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html
        * https://www.cl.cam.ac.uk/~fh295/simlex.html.

        Parameters
        ----------
        pairs : str
            Path to file, where lines are 3-tuples, each consisting of a word pair and a similarity value.
            See `test/test_data/wordsim353.tsv` as example.
        delimiter : str, optional
            Separator in `pairs` file.
        restrict_vocab : int, optional
            Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
            This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
            in modern word embedding models).
        case_insensitive : bool, optional
            If True - convert all words to their uppercase form before evaluating the performance.
            Useful to handle case-mismatch between training tokens and words in the test set.
            In case of multiple case variants of a single word, the vector for the first occurrence
            (also the most frequent if vocabulary is sorted) is taken.
        dummy4unknown : bool, optional
            If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.
            Otherwise, these tuples are skipped entirely and not used in the evaluation.

        Returns
        -------
        pearson : tuple of (float, float)
            Pearson correlation coefficient with 2-tailed p-value.
        spearman : tuple of (float, float)
            Spearman rank-order correlation coefficient between the similarities from the dataset and the
            similarities produced by the model itself, with 2-tailed p-value.
        oov_ratio : float
            The ratio of pairs with unknown words.

        """
    ok_keys = self.index_to_key[:restrict_vocab]
    if case_insensitive:
        ok_vocab = {k.upper(): self.get_index(k) for k in reversed(ok_keys)}
    else:
        ok_vocab = {k: self.get_index(k) for k in reversed(ok_keys)}
    similarity_gold = []
    similarity_model = []
    oov = 0
    original_key_to_index, self.key_to_index = (self.key_to_index, ok_vocab)
    try:
        with utils.open(pairs, encoding=encoding) as fin:
            for line_no, line in enumerate(fin):
                if not line or line.startswith('#'):
                    continue
                try:
                    if case_insensitive:
                        a, b, sim = [word.upper() for word in line.split(delimiter)]
                    else:
                        a, b, sim = [word for word in line.split(delimiter)]
                    sim = float(sim)
                except (ValueError, TypeError):
                    logger.info('Skipping invalid line #%d in %s', line_no, pairs)
                    continue
                if a not in ok_vocab or b not in ok_vocab:
                    oov += 1
                    if dummy4unknown:
                        logger.debug('Zero similarity for line #%d with OOV words: %s', line_no, line.strip())
                        similarity_model.append(0.0)
                        similarity_gold.append(sim)
                    else:
                        logger.info('Skipping line #%d with OOV words: %s', line_no, line.strip())
                    continue
                similarity_gold.append(sim)
                similarity_model.append(self.similarity(a, b))
    finally:
        self.key_to_index = original_key_to_index
    assert len(similarity_gold) == len(similarity_model)
    if not similarity_gold:
        raise ValueError(f'No valid similarity judgements found in {pairs}: either invalid format or all are out-of-vocabulary in {self}')
    spearman = stats.spearmanr(similarity_gold, similarity_model)
    pearson = stats.pearsonr(similarity_gold, similarity_model)
    if dummy4unknown:
        oov_ratio = float(oov) / len(similarity_gold) * 100
    else:
        oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100
    logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
    logger.debug('Spearman rank-order correlation coefficient against %s: %f with p-value %f', pairs, spearman[0], spearman[1])
    logger.debug('Pairs with unknown words: %d', oov)
    self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
    return (pearson, spearman, oov_ratio)