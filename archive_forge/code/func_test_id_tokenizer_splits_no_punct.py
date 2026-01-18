import pytest
@pytest.mark.parametrize('text', ["Ma'arif"])
def test_id_tokenizer_splits_no_punct(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 1