from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
def insertElementNormal(self, token):
    name = token['name']
    assert isinstance(name, text_type), 'Element %s not unicode' % name
    namespace = token.get('namespace', self.defaultNamespace)
    element = self.elementClass(name, namespace)
    element.attributes = token['data']
    self.openElements[-1].appendChild(element)
    self.openElements.append(element)
    return element