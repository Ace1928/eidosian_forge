from collections import defaultdict
from collections.abc import Mapping
import logging
import itertools
from typing import Optional, List, Tuple
from gensim import utils
def filter_n_most_frequent(self, remove_n):
    """Filter out the 'remove_n' most frequent tokens that appear in the documents.

        Parameters
        ----------
        remove_n : int
            Number of the most frequent tokens that will be removed.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import Dictionary
            >>>
            >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
            >>> dct = Dictionary(corpus)
            >>> len(dct)
            5
            >>> dct.filter_n_most_frequent(2)
            >>> len(dct)
            3

        """
    most_frequent_ids = (v for v in self.token2id.values())
    most_frequent_ids = sorted(most_frequent_ids, key=self.dfs.get, reverse=True)
    most_frequent_ids = most_frequent_ids[:remove_n]
    most_frequent_words = [(self[idx], self.dfs.get(idx, 0)) for idx in most_frequent_ids]
    logger.info('discarding %i tokens: %s...', len(most_frequent_ids), most_frequent_words[:10])
    self.filter_tokens(bad_ids=most_frequent_ids)
    logger.info('resulting dictionary: %s', self)