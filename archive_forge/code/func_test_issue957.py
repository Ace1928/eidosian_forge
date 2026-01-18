import pytest
@pytest.mark.issue(957)
@pytest.mark.slow
def test_issue957(en_tokenizer):
    """Test that spaCy doesn't hang on many punctuation characters.
    If this test hangs, check (new) regular expressions for conflicting greedy operators
    """
    pytest.importorskip('pytest_timeout')
    for punct in ['.', ',', "'", '"', ':', '?', '!', ';', '-']:
        string = '0'
        for i in range(1, 100):
            string += punct + str(i)
        doc = en_tokenizer(string)
        assert doc