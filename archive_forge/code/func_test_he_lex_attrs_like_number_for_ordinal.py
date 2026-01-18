import pytest
from spacy.lang.he.lex_attrs import like_num
@pytest.mark.parametrize('word', ['שלישי', 'מליון', 'עשירי', 'מאה', 'עשר', 'אחד עשר'])
def test_he_lex_attrs_like_number_for_ordinal(word):
    assert like_num(word)