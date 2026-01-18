import pytest
def test_en_tokenizer_splits_period_abbr(en_tokenizer):
    text = 'Today is Tuesday.Mr.'
    tokens = en_tokenizer(text)
    assert len(tokens) == 5
    assert tokens[0].text == 'Today'
    assert tokens[1].text == 'is'
    assert tokens[2].text == 'Tuesday'
    assert tokens[3].text == '.'
    assert tokens[4].text == 'Mr.'