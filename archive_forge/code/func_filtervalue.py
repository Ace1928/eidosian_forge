from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
def filtervalue(self, data, attr, value):
    """ Filter to findall occurance of some value in dict """
    items = []
    for item in data:
        if item[attr] == value:
            items.append(item)
    return items