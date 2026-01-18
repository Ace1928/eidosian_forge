import pytest
@pytest.mark.parametrize('text', ["(unter'm"])
def test_de_tokenizer_splits_prefix_punct(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 3