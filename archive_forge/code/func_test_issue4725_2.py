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
@pytest.mark.issue(4725)
def test_issue4725_2():
    if isinstance(get_current_ops, NumpyOps):
        vocab = Vocab(vectors_name='test_vocab_add_vector')
        data = numpy.ndarray((5, 3), dtype='f')
        data[0] = 1.0
        data[1] = 2.0
        vocab.set_vector('cat', data[0])
        vocab.set_vector('dog', data[1])
        nlp = English(vocab=vocab)
        nlp.add_pipe('ner')
        nlp.initialize()
        docs = ['Kurt is in London.'] * 10
        for _ in nlp.pipe(docs, batch_size=2, n_process=2):
            pass