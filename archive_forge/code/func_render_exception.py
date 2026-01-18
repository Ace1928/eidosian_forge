import functools
import re
import wsgiref.util
import http.client
from keystonemiddleware import auth_token
import oslo_i18n
from oslo_log import log
from oslo_serialization import jsonutils
import webob.dec
import webob.exc
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import render_token
from keystone.common import tokenless_auth
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.models import token_model
def render_exception(error, context=None, request=None, user_locale=None):
    """Form a WSGI response based on the current error."""
    error_message = error.args[0]
    message = oslo_i18n.translate(error_message, desired_locale=user_locale)
    if message is error_message:
        message = str(message)
    body = {'error': {'code': error.code, 'title': error.title, 'message': message}}
    headers = []
    if isinstance(error, exception.AuthPluginException):
        body['error']['identity'] = error.authentication
    elif isinstance(error, exception.Unauthorized):
        local_context = {}
        if request:
            local_context = {'environment': request.environ}
        elif context and 'environment' in context:
            local_context = {'environment': context['environment']}
        url = base_url(local_context)
        headers.append(('WWW-Authenticate', 'Keystone uri="%s"' % url))
    return render_response(status=(error.code, error.title), body=body, headers=headers)