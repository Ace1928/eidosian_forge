import pytest
@pytest.mark.parametrize('text,expected_tokens', TESTCASES)
def test_ky_tokenizer_handles_testcases(ky_tokenizer, text, expected_tokens):
    tokens = [token.text for token in ky_tokenizer(text) if not token.is_space]
    assert expected_tokens == tokens