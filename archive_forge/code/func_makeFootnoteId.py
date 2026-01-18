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
def makeFootnoteId(self, id: str) -> str:
    """ Return footnote link id. """
    if self.getConfig('UNIQUE_IDS'):
        return 'fn%s%d-%s' % (self.get_separator(), self.unique_prefix, id)
    else:
        return 'fn{}{}'.format(self.get_separator(), id)