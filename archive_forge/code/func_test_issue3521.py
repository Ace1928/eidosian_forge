import pytest
@pytest.mark.parametrize('word', ["don't", 'don’t', "I'd", 'I’d'])
@pytest.mark.issue(3521)
def test_issue3521(en_tokenizer, word):
    tok = en_tokenizer(word)[1]
    assert tok.is_stop