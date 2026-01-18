from __future__ import absolute_import, unicode_literals
import json
import os
import time
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from oauthlib.signals import scope_changed
from .errors import (InsecureTransportError, MismatchingStateError,
from .tokens import OAuth2Token
from .utils import is_secure_transport, list_to_scope, scope_to_list
def prepare_grant_uri(uri, client_id, response_type, redirect_uri=None, scope=None, state=None, **kwargs):
    """Prepare the authorization grant request URI.

    The client constructs the request URI by adding the following
    parameters to the query component of the authorization endpoint URI
    using the ``application/x-www-form-urlencoded`` format as defined by
    [`W3C.REC-html401-19991224`_]:

    :param uri:
    :param client_id: The client identifier as described in `Section 2.2`_.
    :param response_type: To indicate which OAuth 2 grant/flow is required,
                          "code" and "token".
    :param redirect_uri: The client provided URI to redirect back to after
                         authorization as described in `Section 3.1.2`_.
    :param scope: The scope of the access request as described by
                  `Section 3.3`_.
    :param state: An opaque value used by the client to maintain
                  state between the request and callback.  The authorization
                  server includes this value when redirecting the user-agent
                  back to the client.  The parameter SHOULD be used for
                  preventing cross-site request forgery as described in
                  `Section 10.12`_.
    :param kwargs: Extra arguments to embed in the grant/authorization URL.

    An example of an authorization code grant authorization URL:

    .. code-block:: http

        GET /authorize?response_type=code&client_id=s6BhdRkqt3&state=xyz
            &redirect_uri=https%3A%2F%2Fclient%2Eexample%2Ecom%2Fcb HTTP/1.1
        Host: server.example.com

    .. _`W3C.REC-html401-19991224`:
    https://tools.ietf.org/html/rfc6749#ref-W3C.REC-html401-19991224
    .. _`Section 2.2`: https://tools.ietf.org/html/rfc6749#section-2.2
    .. _`Section 3.1.2`: https://tools.ietf.org/html/rfc6749#section-3.1.2
    .. _`Section 3.3`: https://tools.ietf.org/html/rfc6749#section-3.3
    .. _`section 10.12`: https://tools.ietf.org/html/rfc6749#section-10.12
    """
    if not is_secure_transport(uri):
        raise InsecureTransportError()
    params = [('response_type', response_type), ('client_id', client_id)]
    if redirect_uri:
        params.append(('redirect_uri', redirect_uri))
    if scope:
        params.append(('scope', list_to_scope(scope)))
    if state:
        params.append(('state', state))
    for k in kwargs:
        if kwargs[k]:
            params.append((unicode_type(k), kwargs[k]))
    return add_params_to_uri(uri, params)