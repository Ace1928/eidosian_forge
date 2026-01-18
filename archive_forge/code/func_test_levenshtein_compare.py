import pytest
from spacy.matcher import levenshtein
from spacy.matcher.levenshtein import levenshtein_compare
@pytest.mark.parametrize('a,b,fuzzy,expected', [('a', 'a', 1, True), ('a', 'a', 0, True), ('a', 'a', -1, True), ('a', 'ab', 1, True), ('a', 'ab', 0, False), ('a', 'ab', -1, True), ('ab', 'ac', 1, True), ('ab', 'ac', -1, True), ('abc', 'cde', 4, True), ('abc', 'cde', -1, False), ('abcdef', 'cdefgh', 4, True), ('abcdef', 'cdefgh', 3, False), ('abcdef', 'cdefgh', -1, False), ('abcdefgh', 'cdefghijk', 5, True), ('abcdefgh', 'cdefghijk', 4, False), ('abcdefgh', 'cdefghijk', -1, False), ('abcdefgh', 'cdefghijkl', 6, True), ('abcdefgh', 'cdefghijkl', 5, False), ('abcdefgh', 'cdefghijkl', -1, False)])
def test_levenshtein_compare(a, b, fuzzy, expected):
    assert levenshtein_compare(a, b, fuzzy) == expected