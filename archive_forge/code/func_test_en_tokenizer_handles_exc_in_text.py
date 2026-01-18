import pytest
def test_en_tokenizer_handles_exc_in_text(en_tokenizer):
    text = "It's mediocre i.e. bad."
    tokens = en_tokenizer(text)
    assert len(tokens) == 6
    assert tokens[3].text == 'i.e.'