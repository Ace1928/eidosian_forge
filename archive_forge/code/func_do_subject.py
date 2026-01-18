import copy
import importlib
import logging
import re
from warnings import warn as _warn
from saml2 import saml
from saml2 import xmlenc
from saml2.attribute_converter import ac_factory
from saml2.attribute_converter import from_local
from saml2.attribute_converter import get_local_name
from saml2.s_utils import MissingValue
from saml2.s_utils import assertion_factory
from saml2.s_utils import factory
from saml2.s_utils import sid
from saml2.saml import NAME_FORMAT_URI
from saml2.time_util import in_a_while
from saml2.time_util import instant
def do_subject(not_on_or_after, name_id, **farg):
    specs = farg['subject_confirmation']
    if isinstance(specs, list):
        res = [do_subject_confirmation(not_on_or_after, **s) for s in specs]
    else:
        res = [do_subject_confirmation(not_on_or_after, **specs)]
    return factory(saml.Subject, name_id=name_id, subject_confirmation=res)