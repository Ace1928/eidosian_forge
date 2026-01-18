import pytest
@pytest.mark.parametrize('text,length', [('kerana', 1), ('Mahathir-Anwar', 3), ('Tun Dr. Ismail-Abdul Rahman', 6)])
def test_my_tokenizer_splits_hyphens(ms_tokenizer, text, length):
    tokens = ms_tokenizer(text)
    assert len(tokens) == length