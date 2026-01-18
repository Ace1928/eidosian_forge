import pytest
@pytest.mark.parametrize('text,length', [('ሆሴ ሙጂካ ለምን ተመረጠ?', 5), ('“በፍፁም?”', 4), ('አዎ! ሆዜ አርካዲዮ ቡንዲያ “እንሂድ” ሲል መለሰ።', 11), ('እነሱ በግምት 10ኪ.ሜ. ሮጡ።', 7), ('እና ከዚያ ለምን...', 4)])
def test_am_tokenizer_handles_cnts(am_tokenizer, text, length):
    tokens = am_tokenizer(text)
    assert len(tokens) == length