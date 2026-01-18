import warnings
from bs4.element import (
from . import SoupTest
def test_get_text_ignores_special_string_containers(self):
    soup = self.soup('foo<!--IGNORE-->bar')
    assert soup.get_text() == 'foobar'
    assert soup.get_text(types=(NavigableString, Comment)) == 'fooIGNOREbar'
    assert soup.get_text(types=None) == 'fooIGNOREbar'
    soup = self.soup('foo<style>CSS</style><script>Javascript</script>bar')
    assert soup.get_text() == 'foobar'