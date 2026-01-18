from __future__ import annotations
from . import Extension
from ..inlinepatterns import HtmlInlineProcessor, HTML_RE
from ..treeprocessors import InlineProcessor
from ..util import Registry
from typing import TYPE_CHECKING, Sequence
def educateAngledQuotes(self, md: Markdown) -> None:
    leftAngledQuotePattern = SubstituteTextPattern('\\<\\<', (self.substitutions['left-angle-quote'],), md)
    rightAngledQuotePattern = SubstituteTextPattern('\\>\\>', (self.substitutions['right-angle-quote'],), md)
    self.inlinePatterns.register(leftAngledQuotePattern, 'smarty-left-angle-quotes', 40)
    self.inlinePatterns.register(rightAngledQuotePattern, 'smarty-right-angle-quotes', 35)