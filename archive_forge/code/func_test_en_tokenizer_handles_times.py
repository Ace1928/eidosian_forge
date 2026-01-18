import pytest
@pytest.mark.parametrize('text', ['1am', '12a.m.', '11p.m.', '4pm'])
def test_en_tokenizer_handles_times(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 2