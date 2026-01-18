import pytest
@pytest.mark.parametrize('text', ['km.', 'pvz.', 'biol.'])
def test_lt_tokenizer_abbrev_exceptions(lt_tokenizer, text):
    tokens = lt_tokenizer(text)
    assert len(tokens) == 2