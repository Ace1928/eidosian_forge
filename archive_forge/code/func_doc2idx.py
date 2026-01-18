from collections import defaultdict
from collections.abc import Mapping
import logging
import itertools
from typing import Optional, List, Tuple
from gensim import utils
def doc2idx(self, document, unknown_word_index=-1):
    """Convert `document` (a list of words) into a list of indexes = list of `token_id`.
        Replace all unknown words i.e, words not in the dictionary with the index as set via `unknown_word_index`.

        Parameters
        ----------
        document : list of str
            Input document
        unknown_word_index : int, optional
            Index to use for words not in the dictionary.

        Returns
        -------
        list of int
            Token ids for tokens in `document`, in the same order.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import Dictionary
            >>>
            >>> corpus = [["a", "a", "b"], ["a", "c"]]
            >>> dct = Dictionary(corpus)
            >>> dct.doc2idx(["a", "a", "c", "not_in_dictionary", "c"])
            [0, 0, 2, -1, 2]

        """
    if isinstance(document, str):
        raise TypeError('doc2idx expects an array of unicode tokens on input, not a single string')
    document = [word if isinstance(word, str) else str(word, 'utf-8') for word in document]
    return [self.token2id.get(word, unknown_word_index) for word in document]