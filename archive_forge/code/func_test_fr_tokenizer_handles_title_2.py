import pytest
def test_fr_tokenizer_handles_title_2(fr_tokenizer):
    text = 'Est-ce pas génial?'
    tokens = fr_tokenizer(text)
    assert len(tokens) == 5
    assert tokens[0].text == 'Est'
    assert tokens[1].text == '-ce'