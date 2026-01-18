import contextlib
import re
import tempfile
import numpy
import srsly
from thinc.api import get_current_ops
from spacy.tokens import Doc
from spacy.training import split_bilu_label
from spacy.util import make_tempdir  # noqa: F401
from spacy.vocab import Vocab
def add_vecs_to_vocab(vocab, vectors):
    """Add list of vector tuples to given vocab. All vectors need to have the
    same length. Format: [("text", [1, 2, 3])]"""
    length = len(vectors[0][1])
    vocab.reset_vectors(width=length)
    for word, vec in vectors:
        vocab.set_vector(word, vector=vec)
    return vocab