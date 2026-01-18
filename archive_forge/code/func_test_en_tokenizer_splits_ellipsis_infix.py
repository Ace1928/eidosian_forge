import pytest
@pytest.mark.parametrize('text', ['best...Known', 'best...known'])
def test_en_tokenizer_splits_ellipsis_infix(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 3