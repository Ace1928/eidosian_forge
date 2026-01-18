import hashlib
import json
import os
from django import http
from django import shortcuts
from django.conf import settings
from django.core import urlresolvers
from django.shortcuts import redirect
from django.utils import html
import jsonpickle
from six.moves.urllib import parse
from oauth2client import client
from oauth2client.contrib import django_util
from oauth2client.contrib.django_util import get_storage
from oauth2client.contrib.django_util import signals
def oauth2_callback(request):
    """ View that handles the user's return from OAuth2 provider.

    This view verifies the CSRF state and OAuth authorization code, and on
    success stores the credentials obtained in the storage provider,
    and redirects to the return_url specified in the authorize view and
    stored in the session.

    Args:
        request: Django request.

    Returns:
         A redirect response back to the return_url.
    """
    if 'error' in request.GET:
        reason = request.GET.get('error_description', request.GET.get('error', ''))
        reason = html.escape(reason)
        return http.HttpResponseBadRequest('Authorization failed {0}'.format(reason))
    try:
        encoded_state = request.GET['state']
        code = request.GET['code']
    except KeyError:
        return http.HttpResponseBadRequest('Request missing state or authorization code')
    try:
        server_csrf = request.session[_CSRF_KEY]
    except KeyError:
        return http.HttpResponseBadRequest('No existing session for this flow.')
    try:
        state = json.loads(encoded_state)
        client_csrf = state['csrf_token']
        return_url = state['return_url']
    except (ValueError, KeyError):
        return http.HttpResponseBadRequest('Invalid state parameter.')
    if client_csrf != server_csrf:
        return http.HttpResponseBadRequest('Invalid CSRF token.')
    flow = _get_flow_for_token(client_csrf, request)
    if not flow:
        return http.HttpResponseBadRequest('Missing Oauth2 flow.')
    try:
        credentials = flow.step2_exchange(code)
    except client.FlowExchangeError as exchange_error:
        return http.HttpResponseBadRequest('An error has occurred: {0}'.format(exchange_error))
    get_storage(request).put(credentials)
    signals.oauth2_authorized.send(sender=signals.oauth2_authorized, request=request, credentials=credentials)
    return shortcuts.redirect(return_url)