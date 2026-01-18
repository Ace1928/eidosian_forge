from collections import defaultdict
from collections.abc import Mapping
import logging
import itertools
from typing import Optional, List, Tuple
from gensim import utils
@staticmethod
def from_corpus(corpus, id2word=None):
    """Create :class:`~gensim.corpora.dictionary.Dictionary` from an existing corpus.

        Parameters
        ----------
        corpus : iterable of iterable of (int, number)
            Corpus in BoW format.
        id2word : dict of (int, object)
            Mapping id -> word. If None, the mapping `id2word[word_id] = str(word_id)` will be used.

        Notes
        -----
        This can be useful if you only have a term-document BOW matrix (represented by `corpus`), but not the original
        text corpus. This method will scan the term-document count matrix for all word ids that appear in it,
        then construct :class:`~gensim.corpora.dictionary.Dictionary` which maps each `word_id -> id2word[word_id]`.
        `id2word` is an optional dictionary that maps the `word_id` to a token.
        In case `id2word` isn't specified the mapping `id2word[word_id] = str(word_id)` will be used.

        Returns
        -------
        :class:`~gensim.corpora.dictionary.Dictionary`
            Inferred dictionary from corpus.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import Dictionary
            >>>
            >>> corpus = [[(1, 1.0)], [], [(0, 5.0), (2, 1.0)], []]
            >>> dct = Dictionary.from_corpus(corpus)
            >>> len(dct)
            3

        """
    result = Dictionary()
    max_id = -1
    for docno, document in enumerate(corpus):
        if docno % 10000 == 0:
            logger.info('adding document #%i to %s', docno, result)
        result.num_docs += 1
        result.num_nnz += len(document)
        for wordid, word_freq in document:
            max_id = max(wordid, max_id)
            result.num_pos += word_freq
            result.dfs[wordid] = result.dfs.get(wordid, 0) + 1
    if id2word is None:
        result.token2id = {str(i): i for i in range(max_id + 1)}
    else:
        result.token2id = {utils.to_unicode(token): idx for idx, token in id2word.items()}
    for idx in result.token2id.values():
        result.dfs[idx] = result.dfs.get(idx, 0)
    logger.info('built %s from %i documents (total %i corpus positions)', result, result.num_docs, result.num_pos)
    return result