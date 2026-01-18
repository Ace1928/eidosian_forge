from io import StringIO
from xml.dom import minidom
from xml.etree import ElementTree
from Bio.Phylo import NeXML
from ._cdao_owl import cdao_elements, cdao_namespaces, resolve_uri
def qUri(s):
    """Given a prefixed URI, return the full URI."""
    return resolve_uri(s, namespaces=NAMESPACES, xml_style=True)