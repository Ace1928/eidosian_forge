import pytest
@pytest.mark.parametrize('text', ['Hej,Verden', 'en,to'])
def test_da_tokenizer_splits_comma_infix(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 3
    assert tokens[0].text == text.split(',')[0]
    assert tokens[1].text == ','
    assert tokens[2].text == text.split(',')[1]