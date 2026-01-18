import pytest
def test_fr_tokenizer_handles_title(fr_tokenizer):
    text = "N'est-ce pas génial?"
    tokens = fr_tokenizer(text)
    assert len(tokens) == 6
    assert tokens[0].text == "N'"
    assert tokens[1].text == 'est'
    assert tokens[2].text == '-ce'