import copy
import importlib
import logging
from logging.config import dictConfig as configure_logging_by_dict
import logging.handlers
import os
import re
import sys
from warnings import warn as _warn
from saml2 import BINDING_HTTP_ARTIFACT
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import BINDING_URI
from saml2 import SAMLError
from saml2.assertion import Policy
from saml2.attribute_converter import ac_factory
from saml2.mdstore import MetadataStore
from saml2.saml import NAME_FORMAT_URI
from saml2.virtual_org import VirtualOrg
def config_factory(_type, config):
    """

    :type _type: str
    :param _type:

    :type config: str or dict
    :param config: Name of file with pysaml2 config or CONFIG dict

    :return:
    """
    if _type == 'sp':
        conf = SPConfig()
    elif _type in ['aa', 'idp', 'pdp', 'aq']:
        conf = IdPConfig()
    else:
        conf = Config()
    if isinstance(config, dict):
        conf.load(copy.deepcopy(config))
    elif isinstance(config, str):
        conf.load_file(config)
    else:
        raise ValueError('Unknown type of config')
    conf.context = _type
    return conf