import pytest
@pytest.mark.parametrize('text', ['Jul.', 'jul.', 'Tor.', 'Tors.'])
def test_da_tokenizer_handles_ambiguous_abbr(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 2