import pytest
from spacy import util
from spacy.tokens import Doc
from spacy.vocab import Vocab
def test_create_invalid_pos(vocab):
    words = 'I like ginger'.split()
    pos = 'QQ ZZ XX'.split()
    with pytest.raises(ValueError):
        Doc(vocab, words=words, pos=pos)