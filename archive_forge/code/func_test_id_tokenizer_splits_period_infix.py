import pytest
@pytest.mark.parametrize('text', ['ini.Budi', 'Halo.Bandung'])
def test_id_tokenizer_splits_period_infix(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 3