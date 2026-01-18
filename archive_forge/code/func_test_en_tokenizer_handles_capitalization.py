import pytest
@pytest.mark.parametrize('text_lower,text_title', [("can't", "Can't"), ("ain't", "Ain't")])
def test_en_tokenizer_handles_capitalization(en_tokenizer, text_lower, text_title):
    tokens_lower = en_tokenizer(text_lower)
    tokens_title = en_tokenizer(text_title)
    assert tokens_title[0].text == tokens_lower[0].text.title()
    assert tokens_lower[0].text == tokens_title[0].text.lower()
    assert tokens_lower[1].text == tokens_title[1].text