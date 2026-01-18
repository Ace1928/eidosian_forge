import pytest
@pytest.mark.parametrize('text', ["(can't)"])
def test_en_tokenizer_splits_even_wrap(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 4