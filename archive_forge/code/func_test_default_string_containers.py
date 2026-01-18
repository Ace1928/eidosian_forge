import pytest
from bs4.element import (
from . import SoupTest
def test_default_string_containers(self):
    soup = self.soup('<div>text</div><script>text</script><style>text</style>')
    assert [NavigableString, Script, Stylesheet] == [x.__class__ for x in soup.find_all(string=True)]
    soup = self.soup('<template>Some text<p>In a tag</p></template>Some text outside')
    assert all((isinstance(x, TemplateString) for x in soup.template._all_strings(types=None)))
    outside = soup.template.next_sibling
    assert isinstance(outside, NavigableString)
    assert not isinstance(outside, TemplateString)
    markup = b'<template>Some text<p>In a tag</p><!--with a comment--></template>'
    soup = self.soup(markup)
    assert markup == soup.template.encode('utf8')