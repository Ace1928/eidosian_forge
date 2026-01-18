from html5lib.treebuilders import _base, etree as etree_builders
from lxml import html, etree
def appendChild(self, element):
    self._elementTree.getroot().addnext(element._element)