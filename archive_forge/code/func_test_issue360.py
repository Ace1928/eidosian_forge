import pytest
@pytest.mark.issue(360)
def test_issue360(en_tokenizer):
    """Test tokenization of big ellipsis"""
    tokens = en_tokenizer('$45...............Asking')
    assert len(tokens) > 2