import pytest
@pytest.mark.parametrize('text', ['f.eks.)'])
def test_da_tokenizer_splits_suffix_interact(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[0].text == 'f.eks.'
    assert tokens[1].text == ')'