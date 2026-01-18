from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def parseFragment(file, context, namespaces=True):
    """Parse a fragment of a document, given the context from which it
    was originally extracted.  context should be the parent of the
    node(s) which are in the fragment.

    'file' may be either a file name or an open file object.
    """
    if namespaces:
        builder = FragmentBuilderNS(context)
    else:
        builder = FragmentBuilder(context)
    if isinstance(file, str):
        with open(file, 'rb') as fp:
            result = builder.parseFile(fp)
    else:
        result = builder.parseFile(file)
    return result