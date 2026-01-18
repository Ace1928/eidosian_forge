import sys
from _pydevd_bundle import pydevd_xml
from os.path import basename
from _pydev_bundle import pydev_log
from urllib.parse import unquote_plus
from _pydevd_bundle.pydevd_constants import IS_PY311_OR_GREATER
def print_referrers(obj, stream=None):
    if stream is None:
        stream = sys.stdout
    result = get_referrer_info(obj)
    from xml.dom.minidom import parseString
    dom = parseString(result)
    xml = dom.getElementsByTagName('xml')[0]
    for node in xml.childNodes:
        if node.nodeType == node.TEXT_NODE:
            continue
        if node.localName == 'for':
            stream.write('Searching references for: ')
            for child in node.childNodes:
                if child.nodeType == node.TEXT_NODE:
                    continue
                print_var_node(child, stream)
        elif node.localName == 'var':
            stream.write('Referrer found: ')
            print_var_node(node, stream)
        else:
            sys.stderr.write('Unhandled node: %s\n' % (node,))
    return result