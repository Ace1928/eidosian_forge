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
def handle_duplicates(self, parent: etree.Element) -> None:
    """ Find duplicate footnotes and format and add the duplicates. """
    for li in list(parent):
        count = self.get_num_duplicates(li)
        if count > 1:
            self.add_duplicates(li, count)