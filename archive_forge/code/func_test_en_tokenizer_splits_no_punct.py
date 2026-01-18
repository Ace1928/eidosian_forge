import pytest
@pytest.mark.parametrize('text', ["can't"])
def test_en_tokenizer_splits_no_punct(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 2