import pytest
@pytest.mark.issue(225)
@pytest.mark.xfail(reason='Issue #225 - not yet implemented')
def test_en_tokenizer_splits_em_dash_infix(en_tokenizer):
    tokens = en_tokenizer("Will this road take me to Puddleton?—No, you'll have to walk there.—Ariel.")
    assert tokens[6].text == 'Puddleton'
    assert tokens[7].text == '?'
    assert tokens[8].text == '—'