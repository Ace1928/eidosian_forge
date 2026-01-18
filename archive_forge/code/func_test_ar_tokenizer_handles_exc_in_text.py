import pytest
def test_ar_tokenizer_handles_exc_in_text(ar_tokenizer):
    text = 'تعود الكتابة الهيروغليفية إلى سنة 3200 ق.م'
    tokens = ar_tokenizer(text)
    assert len(tokens) == 7
    assert tokens[6].text == 'ق.م'