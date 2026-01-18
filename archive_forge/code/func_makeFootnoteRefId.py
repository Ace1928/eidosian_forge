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
def makeFootnoteRefId(self, id: str, found: bool=False) -> str:
    """ Return footnote back-link id. """
    if self.getConfig('UNIQUE_IDS'):
        return self.unique_ref('fnref%s%d-%s' % (self.get_separator(), self.unique_prefix, id), found)
    else:
        return self.unique_ref('fnref{}{}'.format(self.get_separator(), id), found)