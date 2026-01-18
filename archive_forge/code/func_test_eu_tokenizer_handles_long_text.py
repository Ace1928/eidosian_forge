import pytest
def test_eu_tokenizer_handles_long_text(eu_tokenizer):
    text = 'ta nere guitarra estrenatu ondoren'
    tokens = eu_tokenizer(text)
    assert len(tokens) == 5