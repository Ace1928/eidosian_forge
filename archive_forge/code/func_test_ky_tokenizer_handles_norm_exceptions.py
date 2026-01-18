import pytest
@pytest.mark.parametrize('text,norms', NORM_TESTCASES)
def test_ky_tokenizer_handles_norm_exceptions(ky_tokenizer, text, norms):
    tokens = ky_tokenizer(text)
    assert [token.norm_ for token in tokens] == norms