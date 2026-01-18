from html5lib.treebuilders import _base, etree as etree_builders
from lxml import html, etree
def insertRoot(self, name):
    buf = []
    if self.doctype and self.doctype.name:
        buf.append('<!DOCTYPE %s' % self.doctype.name)
        if self.doctype.publicId is not None or self.doctype.systemId is not None:
            buf.append(' PUBLIC "%s" "%s"' % (self.doctype.publicId, self.doctype.systemId))
        buf.append('>')
    buf.append('<html></html>')
    root = html.fromstring(''.join(buf))
    for comment in self.initialComments:
        root.addprevious(etree.Comment(comment))
    self.document = self.documentClass()
    self.document._elementTree = root.getroottree()
    root_element = self.elementClass(name)
    root_element._element = root
    self.document.childNodes.append(root_element)
    self.openElements.append(root_element)
    self.rootInserted = True