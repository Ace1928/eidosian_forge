import pytest
@pytest.mark.parametrize('text,expected_tokens', NN_TOKEN_EXCEPTION_TESTS)
def test_nn_tokenizer_handles_exception_cases(nn_tokenizer, text, expected_tokens):
    tokens = nn_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list