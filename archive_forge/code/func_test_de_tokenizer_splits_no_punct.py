import pytest
@pytest.mark.parametrize('text', ["unter'm"])
def test_de_tokenizer_splits_no_punct(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 2