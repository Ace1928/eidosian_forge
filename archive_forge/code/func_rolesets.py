import re
from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
def rolesets(self, baseform=None):
    """
        :return: list of xml descriptions for rolesets.
        """
    if baseform is not None:
        framefile = 'frames/%s.xml' % baseform
        if framefile not in self._framefiles:
            raise ValueError('Frameset file for %s not found' % baseform)
        framefiles = [framefile]
    else:
        framefiles = self._framefiles
    rsets = []
    for framefile in framefiles:
        with self.abspath(framefile).open() as fp:
            etree = ElementTree.parse(fp).getroot()
        rsets.append(etree.findall('predicate/roleset'))
    return LazyConcatenation(rsets)