import pytest
from spacy.lang.hi.lex_attrs import like_num, norm
@pytest.mark.parametrize('word,word_norm', [('चलता', 'चल'), ('पढ़ाई', 'पढ़'), ('देती', 'दे'), ('जाती', 'ज'), ('मुस्कुराकर', 'मुस्कुर')])
def test_hi_norm(word, word_norm):
    assert norm(word) == word_norm