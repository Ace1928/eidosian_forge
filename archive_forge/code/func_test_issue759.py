import pytest
@pytest.mark.issue(759)
@pytest.mark.parametrize('text,is_num', [('one', True), ('ten', True), ('teneleven', False)])
def test_issue759(en_tokenizer, text, is_num):
    tokens = en_tokenizer(text)
    assert tokens[0].like_num == is_num