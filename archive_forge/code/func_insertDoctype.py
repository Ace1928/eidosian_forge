from html5lib.treebuilders import _base, etree as etree_builders
from lxml import html, etree
def insertDoctype(self, name, publicId, systemId):
    doctype = self.doctypeClass(name, publicId, systemId)
    self.doctype = doctype