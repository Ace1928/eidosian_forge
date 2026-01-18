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
def do_attribute_consuming_service(conf, spsso):
    service_description = service_name = None
    requested_attributes = []
    acs = conf.attribute_converters
    req = conf.getattr('required_attributes', 'sp')
    req_attr_name_format = conf.getattr('requested_attribute_name_format', 'sp')
    if req_attr_name_format is None:
        req_attr_name_format = conf.requested_attribute_name_format
    if req:
        requested_attributes.extend(do_requested_attribute(req, acs, is_required='true', name_format=req_attr_name_format))
    opt = conf.getattr('optional_attributes', 'sp')
    if opt:
        requested_attributes.extend(do_requested_attribute(opt, acs, name_format=req_attr_name_format))
    try:
        if conf.description:
            try:
                text, lang = conf.description
            except ValueError:
                text = conf.description
                lang = 'en'
            service_description = [md.ServiceDescription(text=text, lang=lang)]
    except KeyError:
        pass
    try:
        if conf.name:
            try:
                text, lang = conf.name
            except ValueError:
                text = conf.name
                lang = 'en'
            service_name = [md.ServiceName(text=text, lang=lang)]
    except KeyError:
        pass
    if requested_attributes:
        if not service_name:
            service_name = [md.ServiceName(text='', lang='en')]
        ac_serv = md.AttributeConsumingService(index='1', service_name=service_name, requested_attribute=requested_attributes)
        if service_description:
            ac_serv.service_description = service_description
        spsso.attribute_consuming_service = [ac_serv]