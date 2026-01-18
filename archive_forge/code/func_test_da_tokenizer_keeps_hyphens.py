import pytest
@pytest.mark.parametrize('text', ['gå-på-mod', '4-hjulstræk', '100-Pfennig-frimærke', 'TV-2-spots', 'trofæ-vaeggen'])
def test_da_tokenizer_keeps_hyphens(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 1