import pytest
@pytest.mark.parametrize('text', ['(S.Kom.?)'])
def test_id_tokenizer_splits_uneven_wrap_interact(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 4