import pytest
def test_fr_tokenizer_handles_exc_in_text(fr_tokenizer):
    text = 'Je suis allé au mois de janv. aux prud’hommes.'
    tokens = fr_tokenizer(text)
    assert len(tokens) == 10
    assert tokens[6].text == 'janv.'
    assert tokens[8].text == 'prud’hommes'