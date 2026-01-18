import pytest
@pytest.mark.parametrize('text', ["we'll", "You'll", "there'll", "this'll", "those'll"])
def test_en_tokenizer_handles_ll_contraction(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[0].text == text.split("'")[0]
    assert tokens[1].text == "'ll"