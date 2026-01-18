import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def register_prefix(self, nspair):
    """
        Register with ElementTree a set of namespaces

        :param nspair: A dictionary of prefixes and uris to use when
            constructing the text representation.
        :return:
        """
    for prefix, uri in nspair.items():
        try:
            ElementTree.register_namespace(prefix, uri)
        except AttributeError:
            ElementTree._namespace_map[uri] = prefix
        except ValueError:
            pass