from suds import *
from suds.sax.element import Element
def replace_references(self, node):
    """
        Replacing the I{multiref} references with the contents of the
        referenced nodes and remove the I{href} attribute.  Warning:  since
        the I{ref} is not cloned,
        @param node: A node to update.
        @type node: L{Element}
        """
    href = node.getAttribute('href')
    if href is None:
        return
    id = href.getValue()
    ref = self.catalog.get(id)
    if ref is None:
        import logging
        log = logging.getLogger(__name__)
        log.error('soap multiref: %s, not-resolved', id)
        return
    node.append(ref.children)
    node.setText(ref.getText())
    for a in ref.attributes:
        if a.name != 'id':
            node.append(a)
    node.remove(href)