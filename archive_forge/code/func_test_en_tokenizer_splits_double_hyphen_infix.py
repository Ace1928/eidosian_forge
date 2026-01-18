import pytest
def test_en_tokenizer_splits_double_hyphen_infix(en_tokenizer):
    tokens = en_tokenizer('No decent--let alone well-bred--people.')
    assert tokens[0].text == 'No'
    assert tokens[1].text == 'decent'
    assert tokens[2].text == '--'
    assert tokens[3].text == 'let'
    assert tokens[4].text == 'alone'
    assert tokens[5].text == 'well'
    assert tokens[6].text == '-'
    assert tokens[7].text == 'bred'
    assert tokens[8].text == '--'
    assert tokens[9].text == 'people'