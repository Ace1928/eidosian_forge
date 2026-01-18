import pytest
@pytest.mark.parametrize('text', ['best.Known', 'Hello.World'])
def test_en_tokenizer_splits_period_infix(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 3