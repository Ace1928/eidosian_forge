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
@pytest.mark.issue(2871)
def test_issue2871():
    """Test that vectors recover the correct key for spaCy reserved words."""
    words = ['dog', 'cat', 'SUFFIX']
    vocab = Vocab(vectors_name='test_issue2871')
    vocab.vectors.resize(shape=(3, 10))
    vector_data = numpy.zeros((3, 10), dtype='f')
    for word in words:
        _ = vocab[word]
        vocab.set_vector(word, vector_data[0])
    vocab.vectors.name = 'dummy_vectors'
    assert vocab['dog'].rank == 0
    assert vocab['cat'].rank == 1
    assert vocab['SUFFIX'].rank == 2
    assert vocab.vectors.find(key='dog') == 0
    assert vocab.vectors.find(key='cat') == 1
    assert vocab.vectors.find(key='SUFFIX') == 2