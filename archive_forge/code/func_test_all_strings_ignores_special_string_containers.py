import warnings
from bs4.element import (
from . import SoupTest
def test_all_strings_ignores_special_string_containers(self):
    soup = self.soup('foo<!--IGNORE-->bar')
    assert ['foo', 'bar'] == list(soup.strings)
    soup = self.soup('foo<style>CSS</style><script>Javascript</script>bar')
    assert ['foo', 'bar'] == list(soup.strings)