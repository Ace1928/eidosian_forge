import pytest
from spacy.lang.hi.lex_attrs import like_num, norm
@pytest.mark.parametrize('word', ['१९८७', '1987', '१२,२६७', 'उन्नीस', 'पाँच', 'नवासी', '५/१०'])
def test_hi_like_num(word):
    assert like_num(word)