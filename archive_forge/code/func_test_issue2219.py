import numpy
import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine
@pytest.mark.issue(2219)
def test_issue2219(en_vocab):
    """Test if indexing issue still occurs during Token-Token similarity"""
    vectors = [('a', [1, 2, 3]), ('letter', [4, 5, 6])]
    add_vecs_to_vocab(en_vocab, vectors)
    [(word1, vec1), (word2, vec2)] = vectors
    doc = Doc(en_vocab, words=[word1, word2])
    assert doc[0].similarity(doc[1]) == doc[1].similarity(doc[0])