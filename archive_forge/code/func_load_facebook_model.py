import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def load_facebook_model(path, encoding='utf-8'):
    """Load the model from Facebook's native fasttext `.bin` output file.

    Notes
    ------
    Facebook provides both `.vec` and `.bin` files with their modules.
    The former contains human-readable vectors.
    The latter contains machine-readable vectors along with other model parameters.
    This function requires you to **provide the full path to the .bin file**.
    It effectively ignores the `.vec` output file, since it is redundant.

    This function uses the smart_open library to open the path.
    The path may be on a remote host (e.g. HTTP, S3, etc).
    It may also be gzip or bz2 compressed (i.e. end in `.bin.gz` or `.bin.bz2`).
    For details, see `<https://github.com/RaRe-Technologies/smart_open>`__.

    Parameters
    ----------
    model_file : str
        Path to the FastText output files.
        FastText outputs two model files - `/path/to/model.vec` and `/path/to/model.bin`
        Expected value for this example: `/path/to/model` or `/path/to/model.bin`,
        as Gensim requires only `.bin` file to the load entire fastText model.
    encoding : str, optional
        Specifies the file encoding.

    Examples
    --------

    Load, infer, continue training:

    .. sourcecode:: pycon

        >>> from gensim.test.utils import datapath
        >>>
        >>> cap_path = datapath("crime-and-punishment.bin")
        >>> fb_model = load_facebook_model(cap_path)
        >>>
        >>> 'landlord' in fb_model.wv.key_to_index  # Word is out of vocabulary
        False
        >>> oov_term = fb_model.wv['landlord']
        >>>
        >>> 'landlady' in fb_model.wv.key_to_index  # Word is in the vocabulary
        True
        >>> iv_term = fb_model.wv['landlady']
        >>>
        >>> new_sent = [['lord', 'of', 'the', 'rings'], ['lord', 'of', 'the', 'flies']]
        >>> fb_model.build_vocab(new_sent, update=True)
        >>> fb_model.train(sentences=new_sent, total_examples=len(new_sent), epochs=5)

    Returns
    -------
    gensim.models.fasttext.FastText
        The loaded model.

    See Also
    --------
    :func:`~gensim.models.fasttext.load_facebook_vectors` loads
    the word embeddings only.  Its faster, but does not enable you to continue
    training.

    """
    return _load_fasttext_format(path, encoding=encoding, full_model=True)