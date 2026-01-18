import pytest
@pytest.mark.parametrize('text', ["schools'", "Alexis'"])
def test_en_tokenizer_splits_trailing_apos(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[0].text == text.split("'")[0]
    assert tokens[1].text == "'"