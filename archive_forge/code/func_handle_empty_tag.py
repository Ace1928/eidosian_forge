from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..preprocessors import Preprocessor
from ..postprocessors import RawHtmlPostprocessor
from .. import util
from ..htmlparser import HTMLExtractor, blank_line_re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Literal, Mapping
def handle_empty_tag(self, data, is_block):
    if self.inraw or not self.mdstack:
        super().handle_empty_tag(data, is_block)
    elif self.at_line_start() and is_block:
        self.handle_data('\n' + self.md.htmlStash.store(data) + '\n\n')
    else:
        self.handle_data(self.md.htmlStash.store(data))