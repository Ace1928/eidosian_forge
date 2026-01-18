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
def deserialize_subtree(self, xml_doc, bundle):
    """
        Deserialize an etree element containing a PROV document or a bundle
        and write it to the provided internal object.

        :param xml_doc: An etree element containing the information to read.
        :param bundle: The bundle object to write to.
        """
    for element in xml_doc:
        qname = etree.QName(element)
        if qname.namespace != DEFAULT_NAMESPACES['prov'].uri:
            raise ProvXMLException('Non PROV element discovered in document or bundle.')
        if qname.localname == 'other':
            warnings.warn('Document contains non-PROV information in <prov:other>. It will be ignored in this package.', UserWarning)
            continue
        id_tag = _ns_prov('id')
        rec_id = element.attrib[id_tag] if id_tag in element.attrib else None
        if rec_id is not None:
            rec_id = xml_qname_to_QualifiedName(element, rec_id)
        if qname.localname == 'bundleContent':
            b = bundle.bundle(identifier=rec_id)
            self.deserialize_subtree(element, b)
            continue
        attributes = _extract_attributes(element)
        q_prov_name = FULL_PROV_RECORD_IDS_MAP[qname.localname]
        rec_type = PROV_BASE_CLS[q_prov_name]
        if _ns_xsi('type') in element.attrib:
            value = xml_qname_to_QualifiedName(element, element.attrib[_ns_xsi('type')])
            attributes.append((PROV['type'], value))
        rec = bundle.new_record(rec_type, rec_id, attributes)
        if rec_type != q_prov_name:
            rec.add_asserted_type(q_prov_name)
    return bundle