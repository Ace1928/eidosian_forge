import pytest
@pytest.mark.parametrize('text', ['0,1-13,5', '0,0-0,1', '103,27-300', '1/2-3/4'])
def test_da_tokenizer_handles_numeric_range(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 1