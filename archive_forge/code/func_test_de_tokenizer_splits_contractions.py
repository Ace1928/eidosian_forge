import pytest
@pytest.mark.parametrize('text', ["auf'm", "du's", "über'm", "wir's"])
def test_de_tokenizer_splits_contractions(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 2