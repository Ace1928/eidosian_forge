import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def load_facebook_vectors(path, encoding='utf-8'):
    """Load word embeddings from a model saved in Facebook's native fasttext `.bin` format.

    Notes
    ------
    Facebook provides both `.vec` and `.bin` files with their modules.
    The former contains human-readable vectors.
    The latter contains machine-readable vectors along with other model parameters.
    This function requires you to **provide the full path to the .bin file**.
    It effectively ignores the `.vec` output file, since it is redundant.

    This function uses the smart_open library to open the path.
    The path may be on a remote host (e.g. HTTP, S3, etc).
    It may also be gzip or bz2 compressed.
    For details, see `<https://github.com/RaRe-Technologies/smart_open>`__.

    Parameters
    ----------
    path : str
        The location of the model file.
    encoding : str, optional
        Specifies the file encoding.

    Returns
    -------
    gensim.models.fasttext.FastTextKeyedVectors
        The word embeddings.

    Examples
    --------

    Load and infer:

        >>> from gensim.test.utils import datapath
        >>>
        >>> cap_path = datapath("crime-and-punishment.bin")
        >>> fbkv = load_facebook_vectors(cap_path)
        >>>
        >>> 'landlord' in fbkv.key_to_index  # Word is out of vocabulary
        False
        >>> oov_vector = fbkv['landlord']
        >>>
        >>> 'landlady' in fbkv.key_to_index  # Word is in the vocabulary
        True
        >>> iv_vector = fbkv['landlady']

    See Also
    --------
    :func:`~gensim.models.fasttext.load_facebook_model` loads
    the full model, not just word embeddings, and enables you to continue
    model training.

    """
    full_model = _load_fasttext_format(path, encoding=encoding, full_model=False)
    return full_model.wv