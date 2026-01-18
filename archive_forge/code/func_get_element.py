from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..preprocessors import Preprocessor
from ..postprocessors import RawHtmlPostprocessor
from .. import util
from ..htmlparser import HTMLExtractor, blank_line_re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Literal, Mapping
def get_element(self) -> etree.Element:
    """ Return element from `treebuilder` and reset `treebuilder` for later use. """
    element = self.treebuilder.close()
    self.treebuilder = etree.TreeBuilder()
    return element