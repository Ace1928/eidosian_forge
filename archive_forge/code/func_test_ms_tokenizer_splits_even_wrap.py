import pytest
@pytest.mark.parametrize('text', ["(Ma'arif)"])
def test_ms_tokenizer_splits_even_wrap(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 3