import bisect
import random
import numpy as np
from ..io import DataIter, DataBatch, DataDesc
from .. import ndarray
def encode_sentences(sentences, vocab=None, invalid_label=-1, invalid_key='\n', start_label=0, unknown_token=None):
    """Encode sentences and (optionally) build a mapping
    from string tokens to integer indices. Unknown keys
    will be added to vocabulary.

    Parameters
    ----------
    sentences : list of list of str
        A list of sentences to encode. Each sentence
        should be a list of string tokens.
    vocab : None or dict of str -> int
        Optional input Vocabulary
    invalid_label : int, default -1
        Index for invalid token, like <end-of-sentence>
    invalid_key : str, default '\\n'
        Key for invalid token. Use '\\n' for end
        of sentence by default.
    start_label : int
        lowest index.
    unknown_token: str
        Symbol to represent unknown token.
        If not specified, unknown token will be skipped.

    Returns
    -------
    result : list of list of int
        encoded sentences
    vocab : dict of str -> int
        result vocabulary
    """
    idx = start_label
    if vocab is None:
        vocab = {invalid_key: invalid_label}
        new_vocab = True
    else:
        new_vocab = False
    res = []
    for sent in sentences:
        coded = []
        for word in sent:
            if word not in vocab:
                assert new_vocab or unknown_token, 'Unknown token %s' % word
                if idx == invalid_label:
                    idx += 1
                if unknown_token:
                    word = unknown_token
                vocab[word] = idx
                idx += 1
            coded.append(vocab[word])
        res.append(coded)
    return (res, vocab)