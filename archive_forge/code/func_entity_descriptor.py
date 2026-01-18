from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import class_name
from saml2 import md
from saml2 import samlp
from saml2 import xmldsig as ds
from saml2.algsupport import algorithm_support_in_metadata
from saml2.attribute_converter import from_local_name
from saml2.cert import read_cert_from_file
from saml2.config import Config
from saml2.extension import idpdisc
from saml2.extension import mdattr
from saml2.extension import mdui
from saml2.extension import shibmd
from saml2.extension import sp_type
from saml2.md import AttributeProfile
from saml2.s_utils import factory
from saml2.s_utils import rec_factory
from saml2.s_utils import sid
from saml2.saml import NAME_FORMAT_URI
from saml2.saml import Attribute
from saml2.saml import AttributeValue
from saml2.sigver import pre_signature_part
from saml2.sigver import security_context
from saml2.time_util import in_a_while
from saml2.validate import valid_instance
def entity_descriptor(confd):
    mycert = None
    enc_cert = None
    if confd.cert_file is not None:
        mycert = []
        mycert.append(read_cert_from_file(confd.cert_file))
        if confd.additional_cert_files is not None:
            for _cert_file in confd.additional_cert_files:
                mycert.append(read_cert_from_file(_cert_file))
    if confd.encryption_keypairs is not None:
        enc_cert = []
        for _encryption in confd.encryption_keypairs:
            enc_cert.append(read_cert_from_file(_encryption['cert_file']))
    entd = md.EntityDescriptor()
    entd.entity_id = confd.entityid
    if confd.valid_for:
        entd.valid_until = in_a_while(hours=int(confd.valid_for))
    if confd.organization is not None:
        entd.organization = do_organization_info(confd.organization)
    if confd.contact_person is not None:
        entd.contact_person = do_contact_persons_info(confd.contact_person)
    exts = confd.extensions
    if exts:
        if not entd.extensions:
            entd.extensions = md.Extensions()
        for key, val in exts.items():
            _ext = do_extensions(key, val)
            if _ext:
                for _e in _ext:
                    entd.extensions.add_extension_element(_e)
    if confd.entity_attributes:
        if not entd.extensions:
            entd.extensions = md.Extensions()
        attributes = [Attribute(name_format=attr.get('format'), name=attr.get('name'), friendly_name=attr.get('friendly_name'), attribute_value=[AttributeValue(text=value) for value in attr.get('values', [])]) for attr in confd.entity_attributes]
        for attribute in attributes:
            _add_attr_to_entity_attributes(entd.extensions, attribute)
    if confd.assurance_certification:
        if not entd.extensions:
            entd.extensions = md.Extensions()
        ava = [AttributeValue(text=c) for c in confd.assurance_certification]
        attr = Attribute(attribute_value=ava, name='urn:oasis:names:tc:SAML:attribute:assurance-certification')
        _add_attr_to_entity_attributes(entd.extensions, attr)
    if confd.entity_category:
        if not entd.extensions:
            entd.extensions = md.Extensions()
        ava = [AttributeValue(text=c) for c in confd.entity_category]
        attr = Attribute(attribute_value=ava, name='http://macedir.org/entity-category')
        _add_attr_to_entity_attributes(entd.extensions, attr)
    if confd.entity_category_support:
        if not entd.extensions:
            entd.extensions = md.Extensions()
        ava = [AttributeValue(text=c) for c in confd.entity_category_support]
        attr = Attribute(attribute_value=ava, name='http://macedir.org/entity-category-support')
        _add_attr_to_entity_attributes(entd.extensions, attr)
    for item in algorithm_support_in_metadata(confd.xmlsec_binary):
        if not entd.extensions:
            entd.extensions = md.Extensions()
        entd.extensions.add_extension_element(item)
    conf_sp_type = confd.getattr('sp_type', 'sp')
    conf_sp_type_in_md = confd.getattr('sp_type_in_metadata', 'sp')
    if conf_sp_type and conf_sp_type_in_md is True:
        if not entd.extensions:
            entd.extensions = md.Extensions()
        item = sp_type.SPType(text=conf_sp_type)
        entd.extensions.add_extension_element(item)
    serves = confd.serves
    if not serves:
        raise SAMLError('No service type ("sp","idp","aa") provided in the configuration')
    if 'sp' in serves:
        confd.context = 'sp'
        entd.spsso_descriptor = do_spsso_descriptor(confd, mycert, enc_cert)
    if 'idp' in serves:
        confd.context = 'idp'
        entd.idpsso_descriptor = do_idpsso_descriptor(confd, mycert, enc_cert)
    if 'aa' in serves:
        confd.context = 'aa'
        entd.attribute_authority_descriptor = do_aa_descriptor(confd, mycert, enc_cert)
    if 'pdp' in serves:
        confd.context = 'pdp'
        entd.pdp_descriptor = do_pdp_descriptor(confd, mycert, enc_cert)
    if 'aq' in serves:
        confd.context = 'aq'
        entd.authn_authority_descriptor = do_aq_descriptor(confd, mycert, enc_cert)
    return entd