import pytest
@pytest.mark.parametrize('text', ['Halo,Malaysia', 'satu,dua'])
def test_ms_tokenizer_splits_comma_infix(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 3
    assert tokens[0].text == text.split(',')[0]
    assert tokens[1].text == ','
    assert tokens[2].text == text.split(',')[1]