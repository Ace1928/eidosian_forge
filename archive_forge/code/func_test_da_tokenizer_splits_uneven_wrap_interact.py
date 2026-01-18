import pytest
@pytest.mark.parametrize('text', ['(f.eks.?)'])
def test_da_tokenizer_splits_uneven_wrap_interact(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 4
    assert tokens[0].text == '('
    assert tokens[1].text == 'f.eks.'
    assert tokens[2].text == '?'
    assert tokens[3].text == ')'