import pytest
@pytest.mark.parametrize('text,expected_tokens', GRC_TOKEN_EXCEPTION_TESTS)
def test_grc_tokenizer(grc_tokenizer, text, expected_tokens):
    tokens = grc_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list