import re
from lxml import etree, html
def convert_tree(beautiful_soup_tree, makeelement=None):
    """Convert a BeautifulSoup tree to a list of Element trees.

    Returns a list instead of a single root Element to support
    HTML-like soup with more than one root element.

    You can pass a different Element factory through the `makeelement`
    keyword.
    """
    root = _convert_tree(beautiful_soup_tree, makeelement)
    children = root.getchildren()
    for child in children:
        root.remove(child)
    return children