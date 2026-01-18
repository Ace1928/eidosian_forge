import pytest
@pytest.mark.parametrize('text', ['e.g.', 'p.m.', 'Jan.', 'Dec.', 'Inc.'])
def test_en_tokenizer_handles_abbr(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 1