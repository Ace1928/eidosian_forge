from collections import defaultdict
from collections.abc import Mapping
import logging
import itertools
from typing import Optional, List, Tuple
from gensim import utils
def patch_with_special_tokens(self, special_token_dict):
    """Patch token2id and id2token using a dictionary of special tokens.


        **Usecase:** when doing sequence modeling (e.g. named entity recognition), one may  want to specify
        special tokens that behave differently than others.
        One example is the "unknown" token, and another is the padding token.
        It is usual to set the padding token to have index `0`, and patching the dictionary with `{'<PAD>': 0}`
        would be one way to specify this.

        Parameters
        ----------
        special_token_dict : dict of (str, int)
            dict containing the special tokens as keys and their wanted indices as values.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import Dictionary
            >>>
            >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
            >>> dct = Dictionary(corpus)
            >>>
            >>> special_tokens = {'pad': 0, 'space': 1}
            >>> print(dct.token2id)
            {'maso': 0, 'mele': 1, 'máma': 2, 'ema': 3, 'má': 4}
            >>>
            >>> dct.patch_with_special_tokens(special_tokens)
            >>> print(dct.token2id)
            {'maso': 6, 'mele': 7, 'máma': 2, 'ema': 3, 'má': 4, 'pad': 0, 'space': 1}

        """
    possible_ids = []
    for token, idx in special_token_dict.items():
        if token in self.token2id and self.token2id[token] == idx:
            continue
        if token in self.token2id and self.token2id[token] != idx:
            possible_ids.append(self.token2id[token])
            del self.token2id[token]
        old_token = self[idx]
        self.token2id[token] = idx
        self.token2id[old_token] = possible_ids.pop() if len(possible_ids) > 0 else len(self.token2id) - 1
    self.id2token = {}