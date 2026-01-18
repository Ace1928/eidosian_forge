from __future__ import absolute_import, division, unicode_literals
from xml.dom import Node
from ..constants import namespaces, voidElements, spaceCharacters
def emptyTag(self, namespace, name, attrs, hasChildren=False):
    """Generates an EmptyTag token

        :arg namespace: the namespace of the token--can be ``None``

        :arg name: the name of the element

        :arg attrs: the attributes of the element as a dict

        :arg hasChildren: whether or not to yield a SerializationError because
            this tag shouldn't have children

        :returns: EmptyTag token

        """
    yield {'type': 'EmptyTag', 'name': name, 'namespace': namespace, 'data': attrs}
    if hasChildren:
        yield self.error('Void element has children')