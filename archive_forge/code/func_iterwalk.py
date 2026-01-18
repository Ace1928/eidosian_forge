from fontTools.misc.textTools import tostr
def iterwalk(element_or_tree, events=('end',), tag=None):
    """A tree walker that generates events from an existing tree as
        if it was parsing XML data with iterparse().
        Drop-in replacement for lxml.etree.iterwalk.
        """
    if iselement(element_or_tree):
        element = element_or_tree
    else:
        element = element_or_tree.getroot()
    if tag == '*':
        tag = None
    for item in _iterwalk(element, events, tag):
        yield item