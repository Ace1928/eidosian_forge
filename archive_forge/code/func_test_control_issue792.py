import pytest
@pytest.mark.issue(792)
@pytest.mark.parametrize('text', ['This is a string', 'This is a string\n'])
def test_control_issue792(en_tokenizer, text):
    """Test base case for Issue #792: Non-trailing whitespace"""
    doc = en_tokenizer(text)
    assert ''.join([token.text_with_ws for token in doc]) == text