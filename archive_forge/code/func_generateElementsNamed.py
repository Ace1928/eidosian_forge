from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def generateElementsNamed(list, name):
    """Filters Element items in a list with matching name, regardless of URI."""
    for n in list:
        if IElement.providedBy(n) and n.name == name:
            yield n