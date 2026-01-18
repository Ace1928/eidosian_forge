import copy
import logging
import re
from enum import Enum
from botocore import UNSIGNED, xform_name
from botocore.auth import AUTH_TYPE_MAPS, HAS_CRT
from botocore.crt import CRT_SUPPORTED_AUTH_TYPES
from botocore.endpoint_provider import EndpointProvider
from botocore.exceptions import (
from botocore.utils import ensure_boolean, instance_cache
def auth_schemes_to_signing_ctx(self, auth_schemes):
    """Convert an Endpoint's authSchemes property to a signing_context dict

        :type auth_schemes: list
        :param auth_schemes: A list of dictionaries taken from the
            ``authSchemes`` property of an Endpoint object returned by
            ``EndpointProvider``.

        :rtype: str, dict
        :return: Tuple of auth type string (to be used in
            ``request_context['auth_type']``) and signing context dict (for use
            in ``request_context['signing']``).
        """
    if not isinstance(auth_schemes, list) or len(auth_schemes) == 0:
        raise TypeError('auth_schemes must be a non-empty list.')
    LOG.debug('Selecting from endpoint provider\'s list of auth schemes: %s. User selected auth scheme is: "%s"', ', '.join([f'"{s.get('name')}"' for s in auth_schemes]), self._requested_auth_scheme)
    if self._requested_auth_scheme == UNSIGNED:
        return ('none', {})
    auth_schemes = [{**scheme, 'name': self._strip_sig_prefix(scheme['name'])} for scheme in auth_schemes]
    if self._requested_auth_scheme is not None:
        try:
            name, scheme = next(((self._requested_auth_scheme, s) for s in auth_schemes if self._does_botocore_authname_match_ruleset_authname(self._requested_auth_scheme, s['name'])))
        except StopIteration:
            return (None, {})
    else:
        try:
            name, scheme = next(((s['name'], s) for s in auth_schemes if s['name'] in AUTH_TYPE_MAPS))
        except StopIteration:
            fixable_with_crt = False
            auth_type_options = [s['name'] for s in auth_schemes]
            if not HAS_CRT:
                fixable_with_crt = any((scheme in CRT_SUPPORTED_AUTH_TYPES for scheme in auth_type_options))
            if fixable_with_crt:
                raise MissingDependencyException(msg='This operation requires an additional dependency. Use pip install botocore[crt] before proceeding.')
            else:
                raise UnknownSignatureVersionError(signature_version=', '.join(auth_type_options))
    signing_context = {}
    if 'signingRegion' in scheme:
        signing_context['region'] = scheme['signingRegion']
    elif 'signingRegionSet' in scheme:
        if len(scheme['signingRegionSet']) > 0:
            signing_context['region'] = scheme['signingRegionSet'][0]
    if 'signingName' in scheme:
        signing_context.update(signing_name=scheme['signingName'])
    if 'disableDoubleEncoding' in scheme:
        signing_context['disableDoubleEncoding'] = ensure_boolean(scheme['disableDoubleEncoding'])
    LOG.debug('Selected auth type "%s" as "%s" with signing context params: %s', scheme['name'], name, signing_context)
    return (name, signing_context)