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
def do_spsso_descriptor(conf, cert=None, enc_cert=None):
    spsso = md.SPSSODescriptor()
    spsso.protocol_support_enumeration = samlp.NAMESPACE
    exts = conf.getattr('extensions', 'sp')
    if exts:
        if spsso.extensions is None:
            spsso.extensions = md.Extensions()
        for key, val in exts.items():
            _ext = do_extensions(key, val)
            if _ext:
                for _e in _ext:
                    spsso.extensions.add_extension_element(_e)
    endps = conf.getattr('endpoints', 'sp')
    if endps:
        for endpoint, instlist in do_endpoints(endps, ENDPOINTS['sp']).items():
            setattr(spsso, endpoint, instlist)
    ext = do_endpoints(endps, ENDPOINT_EXT['sp'])
    if ext:
        if spsso.extensions is None:
            spsso.extensions = md.Extensions()
        for vals in ext.values():
            for val in vals:
                spsso.extensions.add_extension_element(val)
    ui_info = conf.getattr('ui_info', 'sp')
    if ui_info:
        if spsso.extensions is None:
            spsso.extensions = md.Extensions()
        spsso.extensions.add_extension_element(do_uiinfo(ui_info))
    if cert or enc_cert:
        metadata_key_usage = conf.metadata_key_usage
        spsso.key_descriptor = do_key_descriptor(cert=cert, enc_cert=enc_cert, use=metadata_key_usage)
    for key in ['want_assertions_signed', 'authn_requests_signed']:
        try:
            val = conf.getattr(key, 'sp')
            if val is None:
                setattr(spsso, key, DEFAULT[key])
            else:
                strval = f'{str(val):>s}'
                setattr(spsso, key, strval.lower())
        except KeyError:
            setattr(spsso, key, DEFAULTS[key])
    do_attribute_consuming_service(conf, spsso)
    _do_nameid_format(spsso, conf, 'sp')
    return spsso