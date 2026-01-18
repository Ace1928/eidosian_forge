import pytest
@pytest.mark.parametrize('text', ["d'Saach", "d'Kanner", 'd’Welt', 'd’Suen'])
def test_lb_tokenizer_splits_contractions(lb_tokenizer, text):
    tokens = lb_tokenizer(text)
    assert len(tokens) == 2