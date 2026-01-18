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
@pytest.mark.issue(3412)
def test_issue3412():
    data = numpy.asarray([[0, 0, 0], [1, 2, 3], [9, 8, 7]], dtype='f')
    vectors = Vectors(data=data, keys=['A', 'B', 'C'])
    keys, best_rows, scores = vectors.most_similar(numpy.asarray([[9, 8, 7], [0, 0, 0]], dtype='f'))
    assert best_rows[0] == 2