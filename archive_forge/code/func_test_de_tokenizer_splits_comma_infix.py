import pytest
@pytest.mark.parametrize('text', ['Hallo,Welt', 'eins,zwei'])
def test_de_tokenizer_splits_comma_infix(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 3
    assert tokens[0].text == text.split(',')[0]
    assert tokens[1].text == ','
    assert tokens[2].text == text.split(',')[1]