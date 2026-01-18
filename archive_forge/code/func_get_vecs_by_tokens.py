import logging
import os
import tarfile
import warnings
import zipfile
from . import _constants as C
from . import vocab
from ... import ndarray as nd
from ... import registry
from ... import base
from ...util import is_np_array
from ... import numpy as _mx_np
from ... import numpy_extension as _mx_npx
def get_vecs_by_tokens(self, tokens, lower_case_backup=False):
    """Look up embedding vectors of tokens.


        Parameters
        ----------
        tokens : str or list of strs
            A token or a list of tokens.
        lower_case_backup : bool, default False
            If False, each token in the original case will be looked up; if True, each token in the
            original case will be looked up first, if not found in the keys of the property
            `token_to_idx`, the token in the lower case will be looked up.


        Returns
        -------
        mxnet.ndarray.NDArray:
            The embedding vector(s) of the token(s). According to numpy conventions, if `tokens` is
            a string, returns a 1-D NDArray of shape `self.vec_len`; if `tokens` is a list of
            strings, returns a 2-D NDArray of shape=(len(tokens), self.vec_len).
        """
    to_reduce = False
    if not isinstance(tokens, list):
        tokens = [tokens]
        to_reduce = True
    if not lower_case_backup:
        indices = [self.token_to_idx.get(token, C.UNKNOWN_IDX) for token in tokens]
    else:
        indices = [self.token_to_idx[token] if token in self.token_to_idx else self.token_to_idx.get(token.lower(), C.UNKNOWN_IDX) for token in tokens]
    if is_np_array():
        embedding_fn = _mx_npx.embedding
        array_fn = _mx_np.array
    else:
        embedding_fn = nd.Embedding
        array_fn = nd.array
    vecs = embedding_fn(array_fn(indices), self.idx_to_vec, self.idx_to_vec.shape[0], self.idx_to_vec.shape[1])
    return vecs[0] if to_reduce else vecs