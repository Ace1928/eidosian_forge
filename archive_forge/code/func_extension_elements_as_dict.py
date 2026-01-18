import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def extension_elements_as_dict(extension_elements, onts):
    ees_ = extension_elements_to_elements(extension_elements, onts)
    res = {}
    for elem in ees_:
        try:
            res[elem.c_tag].append(elem)
        except KeyError:
            res[elem.c_tag] = [elem]
    return res