import datetime
import logging
from lxml import etree
import io
import warnings
import prov
import prov.identifier
from prov.model import DEFAULT_NAMESPACES, sorted_attributes
from prov.constants import *  # NOQA
from prov.serializers import Serializer
def serialize_bundle(self, bundle, element=None, force_types=False):
    """
        Serializes a bundle or document to PROV XML.

        :param bundle: The bundle or document.
        :param element: The XML element to write to. Will be created if None.
        :type force_types: boolean, optional
        :param force_types: Will force xsd:types to be written for most
            attributes mainly PROV-"attributes", e.g. tags not in the
            PROV namespace. Off by default meaning xsd:type attributes will
            only be set for prov:type, prov:location, and prov:value as is
            done in the official PROV-XML specification. Furthermore the
            types will always be set if the Python type requires it. False
            is a good default and it should rarely require changing.
        """
    nsmap = {ns.prefix: ns.uri for ns in self.document._namespaces.get_registered_namespaces()}
    if self.document._namespaces._default:
        nsmap[None] = self.document._namespaces._default.uri
    for namespace in bundle.namespaces:
        if namespace not in nsmap:
            nsmap[namespace.prefix] = namespace.uri
    for key, value in DEFAULT_NAMESPACES.items():
        uri = value.uri
        if value.prefix == 'xsd':
            uri = uri.rstrip('#')
        nsmap[value.prefix] = uri
    if element is not None:
        xml_bundle_root = etree.SubElement(element, _ns_prov('bundleContent'), nsmap=nsmap)
    else:
        xml_bundle_root = etree.Element(_ns_prov('document'), nsmap=nsmap)
    if bundle.identifier:
        xml_bundle_root.attrib[_ns_prov('id')] = str(bundle.identifier)
    for record in bundle._records:
        rec_type = record.get_type()
        identifier = str(record._identifier) if record._identifier else None
        if identifier:
            attrs = {_ns_prov('id'): identifier}
        else:
            attrs = None
        attributes = list(record.attributes)
        rec_label = self._derive_record_label(rec_type, attributes)
        elem = etree.SubElement(xml_bundle_root, _ns_prov(rec_label), attrs)
        for attr, value in sorted_attributes(rec_type, attributes):
            subelem = etree.SubElement(elem, _ns(attr.namespace.uri, attr.localpart))
            if isinstance(value, prov.model.Literal):
                if value.datatype not in [None, PROV['InternationalizedString']]:
                    subelem.attrib[_ns_xsi('type')] = '%s:%s' % (value.datatype.namespace.prefix, value.datatype.localpart)
                if value.langtag is not None:
                    subelem.attrib[_ns_xml('lang')] = value.langtag
                v = value.value
            elif isinstance(value, prov.model.QualifiedName):
                if attr not in PROV_ATTRIBUTE_QNAMES:
                    subelem.attrib[_ns_xsi('type')] = 'xsd:QName'
                v = str(value)
            elif isinstance(value, datetime.datetime):
                v = value.isoformat()
            else:
                v = str(value)
            ALWAYS_CHECK = [bool, datetime.datetime, float, int, prov.identifier.Identifier]
            ALWAYS_CHECK = tuple(ALWAYS_CHECK)
            if (force_types or type(value) in ALWAYS_CHECK or attr in [PROV_TYPE, PROV_LOCATION, PROV_VALUE]) and _ns_xsi('type') not in subelem.attrib and (not str(value).startswith('prov:')) and (not (attr in PROV_ATTRIBUTE_QNAMES and v)) and (attr not in [PROV_ATTR_TIME, PROV_LABEL]):
                xsd_type = None
                if isinstance(value, bool):
                    xsd_type = XSD_BOOLEAN
                    v = v.lower()
                elif isinstance(value, str):
                    xsd_type = XSD_STRING
                elif isinstance(value, float):
                    xsd_type = XSD_DOUBLE
                elif isinstance(value, int):
                    xsd_type = XSD_INT
                elif isinstance(value, datetime.datetime):
                    if attr.namespace.prefix != 'prov' or 'time' not in attr.localpart.lower():
                        xsd_type = XSD_DATETIME
                elif isinstance(value, prov.identifier.Identifier):
                    xsd_type = XSD_ANYURI
                if xsd_type is not None:
                    subelem.attrib[_ns_xsi('type')] = str(xsd_type)
            if attr in PROV_ATTRIBUTE_QNAMES and v:
                subelem.attrib[_ns_prov('ref')] = v
            else:
                subelem.text = v
    return xml_bundle_root