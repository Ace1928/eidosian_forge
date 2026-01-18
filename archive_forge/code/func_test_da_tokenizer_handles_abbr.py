import pytest
@pytest.mark.parametrize('text', ['ca.', 'm.a.o.', 'Jan.', 'Dec.', 'kr.', 'jf.'])
def test_da_tokenizer_handles_abbr(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 1