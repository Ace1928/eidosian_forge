from io import StringIO
from xml.dom import minidom
from xml.etree import ElementTree
from Bio.Phylo import NeXML
from ._cdao_owl import cdao_elements, cdao_namespaces, resolve_uri
def cdao_to_obo(s):
    """Optionally converts a CDAO-prefixed URI into an OBO-prefixed URI."""
    return f'obo:{cdao_elements[s[len('cdao:'):]]}'