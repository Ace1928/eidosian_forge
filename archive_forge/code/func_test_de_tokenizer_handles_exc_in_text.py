import pytest
def test_de_tokenizer_handles_exc_in_text(de_tokenizer):
    text = 'Ich bin z.Zt. im Urlaub.'
    tokens = de_tokenizer(text)
    assert len(tokens) == 6
    assert tokens[2].text == 'z.Zt.'