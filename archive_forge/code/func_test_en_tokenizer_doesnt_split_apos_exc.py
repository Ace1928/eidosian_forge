import pytest
@pytest.mark.parametrize('text', ["'em", "nothin'", "ol'"])
def test_en_tokenizer_doesnt_split_apos_exc(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 1
    assert tokens[0].text == text