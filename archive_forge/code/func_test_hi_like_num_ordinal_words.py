import pytest
from spacy.lang.hi.lex_attrs import like_num, norm
@pytest.mark.parametrize('word', ['पहला', 'तृतीय', 'निन्यानवेवाँ', 'उन्नीस', 'तिहत्तरवाँ', 'छत्तीसवाँ'])
def test_hi_like_num_ordinal_words(word):
    assert like_num(word)