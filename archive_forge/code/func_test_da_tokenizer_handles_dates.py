import pytest
@pytest.mark.parametrize('text', ['1.', '10.', '31.'])
def test_da_tokenizer_handles_dates(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 1