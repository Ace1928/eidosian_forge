import pytest
def test_da_tokenizer_splits_double_hyphen_infix(da_tokenizer):
    tokens = da_tokenizer('Mange regler--eksempelvis bindestregs-reglerne--er komplicerede.')
    assert len(tokens) == 9
    assert tokens[0].text == 'Mange'
    assert tokens[1].text == 'regler'
    assert tokens[2].text == '--'
    assert tokens[3].text == 'eksempelvis'
    assert tokens[4].text == 'bindestregs-reglerne'
    assert tokens[5].text == '--'
    assert tokens[6].text == 'er'
    assert tokens[7].text == 'komplicerede'