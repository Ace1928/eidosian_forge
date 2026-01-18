import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def loadd(self, ava, base64encode=False):
    """
        Sets attributes, children, extension elements and extension
        attributes of this element instance depending on what is in
        the given dictionary. If there are already values on properties
        those will be overwritten. If the keys in the dictionary does
        not correspond to known attributes/children/.. they are ignored.

        :param ava: The dictionary
        :param base64encode: Whether the values on attributes or texts on
            children shoule be base64encoded.
        :return: The instance
        """
    for prop, _typ, _req in self.c_attributes.values():
        if prop in ava:
            value = ava[prop]
            if isinstance(value, (bool, int)):
                setattr(self, prop, str(value))
            else:
                setattr(self, prop, value)
    if 'text' in ava:
        self.set_text(ava['text'], base64encode)
    for prop, klassdef in self.c_children.values():
        if prop in ava:
            if isinstance(klassdef, list):
                make_vals(ava[prop], klassdef[0], self, prop, base64encode=base64encode)
            else:
                cis = make_vals(ava[prop], klassdef, self, prop, True, base64encode)
                setattr(self, prop, cis)
    if 'extension_elements' in ava:
        for item in ava['extension_elements']:
            self.extension_elements.append(ExtensionElement(item['tag']).loadd(item))
    if 'extension_attributes' in ava:
        for key, val in ava['extension_attributes'].items():
            self.extension_attributes[key] = val
    return self