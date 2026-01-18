import pytest
@pytest.mark.parametrize('text', ['ini.Sani', 'Halo.Malaysia'])
def test_ms_tokenizer_splits_period_infix(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 3