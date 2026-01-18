from suds import *
from suds.sax.element import Element
def build_catalog(self, body):
    """
        Create the I{catalog} of multiref nodes by id and the list of
        non-multiref nodes.
        @param body: A soap envelope body node.
        @type body: L{Element}
        """
    for child in body.children:
        if self.soaproot(child):
            self.nodes.append(child)
        id = child.get('id')
        if id is None:
            continue
        key = '#%s' % id
        self.catalog[key] = child