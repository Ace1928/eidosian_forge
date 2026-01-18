import re
from lxml import etree
from .jsonutil import JsonTable
def element_attrs(self, name):
    """ Returns the attributes of this specific element.
        """
    attrs = []
    for element in self.__call__('//%s' % name):
        attrs.append(element.attrib)
    return attrs