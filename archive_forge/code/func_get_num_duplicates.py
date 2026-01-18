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
def get_num_duplicates(self, li: etree.Element) -> int:
    """ Get the number of duplicate refs of the footnote. """
    fn, rest = li.attrib.get('id', '').split(self.footnotes.get_separator(), 1)
    link_id = '{}ref{}{}'.format(fn, self.footnotes.get_separator(), rest)
    return self.footnotes.found_refs.get(link_id, 0)