import pytest
@pytest.mark.parametrize('text,length', [('S.Kom.', 1), ('SKom.', 2), ('(S.Kom.', 2)])
def test_ms_tokenizer_splits_prefix_interact(id_tokenizer, text, length):
    tokens = id_tokenizer(text)
    assert len(tokens) == length