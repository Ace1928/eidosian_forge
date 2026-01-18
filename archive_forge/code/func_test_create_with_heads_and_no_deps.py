import pytest
from spacy import util
from spacy.tokens import Doc
from spacy.vocab import Vocab
def test_create_with_heads_and_no_deps(vocab):
    words = 'I like ginger'.split()
    heads = list(range(len(words)))
    with pytest.raises(ValueError):
        Doc(vocab, words=words, heads=heads)