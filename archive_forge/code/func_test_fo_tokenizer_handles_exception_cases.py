import pytest
@pytest.mark.parametrize('text,expected_tokens', FO_TOKEN_EXCEPTION_TESTS)
def test_fo_tokenizer_handles_exception_cases(fo_tokenizer, text, expected_tokens):
    tokens = fo_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list