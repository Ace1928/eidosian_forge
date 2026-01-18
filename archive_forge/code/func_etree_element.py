from collections import namedtuple
from collections.abc import MutableMapping, MutableSequence
from typing import TYPE_CHECKING, cast, Any, Dict, Iterator, Iterable, \
from xml.etree.ElementTree import Element
from ..exceptions import XMLSchemaTypeError
from ..names import XSI_NAMESPACE
from ..aliases import NamespacesType, BaseXsdType
from ..namespaces import NamespaceMapper
def etree_element(self, tag: str, text: Optional[str]=None, children: Optional[List[Element]]=None, attrib: Optional[Dict[str, str]]=None, level: int=0) -> Element:
    """
        Builds an ElementTree's Element using arguments and the element class and
        the indent spacing stored in the converter instance.

        :param tag: the Element tag string.
        :param text: the Element text.
        :param children: the list of Element children/subelements.
        :param attrib: a dictionary with Element attributes.
        :param level: the level related to the encoding process (0 means the root).
        :return: an instance of the Element class is set for the converter instance.
        """
    if type(self.etree_element_class) is type(Element):
        if attrib is None:
            elem = self.etree_element_class(tag)
        else:
            elem = self.etree_element_class(tag, self.dict(attrib))
    else:
        nsmap = {prefix if prefix else None: uri for prefix, uri in self._namespaces.items() if uri}
        elem = self.etree_element_class(tag, nsmap=nsmap)
        elem.attrib.update(attrib)
    if children:
        elem.extend(children)
        elem.text = text or '\n' + ' ' * self.indent * (level + 1)
        elem.tail = '\n' + ' ' * self.indent * level
    else:
        elem.text = text
        elem.tail = '\n' + ' ' * self.indent * level
    return elem