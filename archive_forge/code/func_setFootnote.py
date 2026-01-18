from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..treeprocessors import Treeprocessor
from ..postprocessors import Postprocessor
from .. import util
from collections import OrderedDict
import re
import copy
import xml.etree.ElementTree as etree
def setFootnote(self, id: str, text: str) -> None:
    """ Store a footnote for later retrieval. """
    self.footnotes[id] = text