import pytest
@pytest.mark.parametrize('text', ["ta'r", "Søren's", "Lars'"])
def test_da_tokenizer_handles_no_punct(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 1