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
def do_subject_confirmation(not_on_or_after, key_info=None, **treeargs):
    """

    :param not_on_or_after: not_on_or_after policy
    :param subject_confirmation_method: How was the subject confirmed
    :param address: The network address/location from which an attesting entity
        can present the assertion.
    :param key_info: Information of the key used to confirm the subject
    :param in_response_to: The ID of a SAML protocol message in response to
        which an attesting entity can present the assertion.
    :param recipient: A URI specifying the entity or location to which an
        attesting entity can present the assertion.
    :param not_before: A time instant before which the subject cannot be
        confirmed. The time value MUST be encoded in UTC.
    :return:
    """
    _sc = factory(saml.SubjectConfirmation, **treeargs)
    _scd = _sc.subject_confirmation_data
    _scd.not_on_or_after = not_on_or_after
    if _sc.method == saml.SCM_HOLDER_OF_KEY:
        _scd.add_extension_element(key_info)
    return _sc