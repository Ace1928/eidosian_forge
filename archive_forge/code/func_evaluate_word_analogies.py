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
def evaluate_word_analogies(self, analogies, restrict_vocab=300000, case_insensitive=True, dummy4unknown=False, similarity_function='most_similar'):
    """Compute performance of the model on an analogy test set.

        The accuracy is reported (printed to log and returned as a score) for each section separately,
        plus there's one aggregate summary at the end.

        This method corresponds to the `compute-accuracy` script of the original C word2vec.
        See also `Analogy (State of the art) <https://aclweb.org/aclwiki/Analogy_(State_of_the_art)>`_.

        Parameters
        ----------
        analogies : str
            Path to file, where lines are 4-tuples of words, split into sections by ": SECTION NAME" lines.
            See `gensim/test/test_data/questions-words.txt` as example.
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
        similarity_function : str, optional
            Function name used for similarity calculation.

        Returns
        -------
        score : float
            The overall evaluation score on the entire evaluation set
        sections : list of dict of {str : str or list of tuple of (str, str, str, str)}
            Results broken down by each section of the evaluation set. Each dict contains the name of the section
            under the key 'section', and lists of correctly and incorrectly predicted 4-tuples of words under the
            keys 'correct' and 'incorrect'.

        """
    ok_keys = self.index_to_key[:restrict_vocab]
    if case_insensitive:
        ok_vocab = {k.upper(): self.get_index(k) for k in reversed(ok_keys)}
    else:
        ok_vocab = {k: self.get_index(k) for k in reversed(ok_keys)}
    oov = 0
    logger.info('Evaluating word analogies for top %i words in the model on %s', restrict_vocab, analogies)
    sections, section = ([], None)
    quadruplets_no = 0
    with utils.open(analogies, 'rb') as fin:
        for line_no, line in enumerate(fin):
            line = utils.to_unicode(line)
            if line.startswith(': '):
                if section:
                    sections.append(section)
                    self._log_evaluate_word_analogies(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError('Missing section header before line #%i in %s' % (line_no, analogies))
                try:
                    if case_insensitive:
                        a, b, c, expected = [word.upper() for word in line.split()]
                    else:
                        a, b, c, expected = [word for word in line.split()]
                except ValueError:
                    logger.info('Skipping invalid line #%i in %s', line_no, analogies)
                    continue
                quadruplets_no += 1
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or (expected not in ok_vocab):
                    oov += 1
                    if dummy4unknown:
                        logger.debug('Zero accuracy for line #%d with OOV words: %s', line_no, line.strip())
                        section['incorrect'].append((a, b, c, expected))
                    else:
                        logger.debug('Skipping line #%i with OOV words: %s', line_no, line.strip())
                    continue
                original_key_to_index = self.key_to_index
                self.key_to_index = ok_vocab
                ignore = {a, b, c}
                predicted = None
                sims = self.most_similar(positive=[b, c], negative=[a], topn=5, restrict_vocab=restrict_vocab)
                self.key_to_index = original_key_to_index
                for element in sims:
                    predicted = element[0].upper() if case_insensitive else element[0]
                    if predicted in ok_vocab and predicted not in ignore:
                        if predicted != expected:
                            logger.debug('%s: expected %s, predicted %s', line.strip(), expected, predicted)
                        break
                if predicted == expected:
                    section['correct'].append((a, b, c, expected))
                else:
                    section['incorrect'].append((a, b, c, expected))
    if section:
        sections.append(section)
        self._log_evaluate_word_analogies(section)
    total = {'section': 'Total accuracy', 'correct': list(itertools.chain.from_iterable((s['correct'] for s in sections))), 'incorrect': list(itertools.chain.from_iterable((s['incorrect'] for s in sections)))}
    oov_ratio = float(oov) / quadruplets_no * 100
    logger.info('Quadruplets with out-of-vocabulary words: %.1f%%', oov_ratio)
    if not dummy4unknown:
        logger.info('NB: analogies containing OOV words were skipped from evaluation! To change this behavior, use "dummy4unknown=True"')
    analogies_score = self._log_evaluate_word_analogies(total)
    sections.append(total)
    return (analogies_score, sections)