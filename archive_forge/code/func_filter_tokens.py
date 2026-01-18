from collections import defaultdict
from collections.abc import Mapping
import logging
import itertools
from typing import Optional, List, Tuple
from gensim import utils
def filter_tokens(self, bad_ids=None, good_ids=None):
    """Remove the selected `bad_ids` tokens from :class:`~gensim.corpora.dictionary.Dictionary`.

        Alternatively, keep selected `good_ids` in :class:`~gensim.corpora.dictionary.Dictionary` and remove the rest.

        Parameters
        ----------
        bad_ids : iterable of int, optional
            Collection of word ids to be removed.
        good_ids : collection of int, optional
            Keep selected collection of word ids and remove the rest.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import Dictionary
            >>>
            >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
            >>> dct = Dictionary(corpus)
            >>> 'ema' in dct.token2id
            True
            >>> dct.filter_tokens(bad_ids=[dct.token2id['ema']])
            >>> 'ema' in dct.token2id
            False
            >>> len(dct)
            4
            >>> dct.filter_tokens(good_ids=[dct.token2id['maso']])
            >>> len(dct)
            1

        """
    if bad_ids is not None:
        bad_ids = set(bad_ids)
        self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid not in bad_ids}
        self.cfs = {tokenid: freq for tokenid, freq in self.cfs.items() if tokenid not in bad_ids}
        self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid not in bad_ids}
    if good_ids is not None:
        good_ids = set(good_ids)
        self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid in good_ids}
        self.cfs = {tokenid: freq for tokenid, freq in self.cfs.items() if tokenid in good_ids}
        self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid in good_ids}
    self.compactify()