import pytest
@pytest.mark.parametrize('text,expected', [("(ta'r)", ['(', "ta'r", ')']), ("'ta'r'", ["'", "ta'r", "'"])])
def test_da_tokenizer_splits_even_wrap(da_tokenizer, text, expected):
    tokens = da_tokenizer(text)
    assert len(tokens) == len(expected)
    assert [t.text for t in tokens] == expected