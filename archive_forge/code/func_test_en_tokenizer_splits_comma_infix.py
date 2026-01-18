import pytest
@pytest.mark.parametrize('text', ['Hello,world', 'one,two'])
def test_en_tokenizer_splits_comma_infix(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 3
    assert tokens[0].text == text.split(',')[0]
    assert tokens[1].text == ','
    assert tokens[2].text == text.split(',')[1]