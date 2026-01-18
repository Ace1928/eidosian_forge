from importlib import import_module
import logging
import os
import sys
from saml2 import NAMESPACE
from saml2 import ExtensionElement
from saml2 import SAMLError
from saml2 import extension_elements_to_elements
from saml2 import saml
from saml2.s_utils import do_ava
from saml2.s_utils import factory
from saml2.saml import NAME_FORMAT_UNSPECIFIED
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def load_maps(dirspec):
    """load the attribute maps

    :param dirspec: a directory specification
    :return: a dictionary with the name of the map as key and the
        map as value. The map itself is a dictionary with two keys:
        "to" and "fro". The values for those keys are the actual mapping.
    """
    mapd = {}
    if dirspec not in sys.path:
        sys.path.insert(0, dirspec)
    for fil in os.listdir(dirspec):
        if fil.endswith('.py'):
            mod = import_module(fil[:-3])
            for item in _find_maps_in_module(mod):
                mapd[item['identifier']] = item
    return mapd