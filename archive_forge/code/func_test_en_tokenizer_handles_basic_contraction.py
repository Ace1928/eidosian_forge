import pytest
def test_en_tokenizer_handles_basic_contraction(en_tokenizer):
    text = "don't giggle"
    tokens = en_tokenizer(text)
    assert len(tokens) == 3
    assert tokens[1].text == "n't"
    text = "i said don't!"
    tokens = en_tokenizer(text)
    assert len(tokens) == 5
    assert tokens[4].text == '!'