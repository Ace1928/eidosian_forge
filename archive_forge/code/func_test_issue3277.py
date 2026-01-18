import pytest
@pytest.mark.issue(3277)
def test_issue3277(es_tokenizer):
    """Test that hyphens are split correctly as prefixes."""
    doc = es_tokenizer('—Yo me llamo... –murmuró el niño– Emilio Sánchez Pérez.')
    assert len(doc) == 14
    assert doc[0].text == '—'
    assert doc[5].text == '–'
    assert doc[9].text == '–'