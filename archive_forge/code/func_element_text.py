import re
from lxml import etree
from .jsonutil import JsonTable
def element_text(self, name):
    """ Returns the text values of this specific element.
        """
    return list(set(self.__call__('//%s/child::text()' % name)))