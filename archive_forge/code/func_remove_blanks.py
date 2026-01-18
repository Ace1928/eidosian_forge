import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
def remove_blanks(elem):
    """
    Remove all elements and subelements with no text and no child elements.

    :param elem: toolbox data in an elementtree structure
    :type elem: ElementTree._ElementInterface
    """
    out = list()
    for child in elem:
        remove_blanks(child)
        if child.text or len(child) > 0:
            out.append(child)
    elem[:] = out