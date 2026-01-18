import numpy
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.lang.en import English
from spacy.strings import hash_string  # type: ignore
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training.initialize import convert_vectors
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine, make_tempdir
def test_init_vectors_with_resize_shape(strings, resize_data):
    v = Vectors(shape=(len(strings), 3))
    v.resize(shape=resize_data.shape)
    assert v.shape == resize_data.shape
    assert v.shape != (len(strings), 3)