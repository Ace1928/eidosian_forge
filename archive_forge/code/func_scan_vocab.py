import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer
from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
def scan_vocab(self, corpus_iterable=None, corpus_file=None, progress_per=100000, trim_rule=None):
    """Create the model's vocabulary: a mapping from unique words in the corpus to their frequency count.

        Parameters
        ----------
        documents : iterable of :class:`~gensim.models.doc2vec.TaggedDocument`, optional
            The tagged documents used to create the vocabulary. Their tags can be either str tokens or ints (faster).
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `documents` to get performance boost. Only one of `documents` or
            `corpus_file` arguments need to be passed (not both of them).
        progress_per : int
            Progress will be logged every `progress_per` documents.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.doc2vec.Doc2Vec.build_vocab` and is not stored as part of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        Returns
        -------
        (int, int)
            Tuple of `(total words in the corpus, number of documents)`.

        """
    logger.info('collecting all words and their counts')
    if corpus_file is not None:
        corpus_iterable = TaggedLineDocument(corpus_file)
    total_words, corpus_count = self._scan_vocab(corpus_iterable, progress_per, trim_rule)
    logger.info('collected %i word types and %i unique tags from a corpus of %i examples and %i words', len(self.raw_vocab), len(self.dv), corpus_count, total_words)
    return (total_words, corpus_count)