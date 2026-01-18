import pytest
@pytest.mark.parametrize('text', ['sort.Gul', 'Hej.Verden'])
def test_da_tokenizer_splits_period_infix(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 3