import warnings
from bs4.element import (
from . import SoupTest
def test__should_pretty_print(self):
    tag = self.soup('').new_tag('a_tag')
    tag._preserve_whitespace_tags = None
    assert True == tag._should_pretty_print(0)
    tag.preserve_whitespace_tags = ['some_other_tag']
    assert True == tag._should_pretty_print(1)
    assert False == tag._should_pretty_print(None)
    tag.preserve_whitespace_tags = ['some_other_tag', 'a_tag']
    assert False == tag._should_pretty_print(1)