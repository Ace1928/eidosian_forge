import re
from lxml import etree
from .jsonutil import JsonTable
def element_values(self, name, key):
    """ Returns the attribute values of this specific element.
        """
    values = set()
    for subject in self.__call__('//%s' % name):
        values.add(subject.get(key))
    return list(values)