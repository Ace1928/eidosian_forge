import pytest
@pytest.mark.parametrize('text,length', [('_MATH_', 3), ('_MATH_.', 4)])
def test_final_period(en_tokenizer, text, length):
    tokens = en_tokenizer(text)
    assert len(tokens) == length