import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.parametrize('text,inflections,reading_forms', [('取ってつけた', (['五段-ラ行;連用形-促音便'], [], ['下一段-カ行;連用形-一般'], ['助動詞-タ;終止形-一般']), (['トッ'], ['テ'], ['ツケ'], ['タ'])), ('2=3', ([], [], []), (['ニ'], ['_'], ['サン']))])
def test_ja_tokenizer_inflections_reading_forms(ja_tokenizer, text, inflections, reading_forms):
    tokens = ja_tokenizer(text)
    test_inflections = [tt.morph.get('Inflection') for tt in tokens]
    assert test_inflections == list(inflections)
    test_readings = [tt.morph.get('Reading') for tt in tokens]
    assert test_readings == list(reading_forms)