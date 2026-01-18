import pytest
@pytest.mark.parametrize('exc', ['Ill', 'ill', 'Hell', 'hell', 'Well', 'well'])
def test_en_tokenizer_excludes_ambiguous(en_tokenizer, exc):
    tokens = en_tokenizer(exc)
    assert len(tokens) == 1