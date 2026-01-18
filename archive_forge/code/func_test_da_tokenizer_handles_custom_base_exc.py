import pytest
def test_da_tokenizer_handles_custom_base_exc(da_tokenizer):
    text = 'Her er noget du kan kigge i.'
    tokens = da_tokenizer(text)
    assert len(tokens) == 8
    assert tokens[6].text == 'i'
    assert tokens[7].text == '.'