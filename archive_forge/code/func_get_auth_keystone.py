import socket
import re
import logging
import warnings
from requests.exceptions import RequestException, SSLError
import http.client as http_client
from urllib.parse import quote, unquote
from urllib.parse import urljoin, urlparse, urlunparse
from time import sleep, time
from swiftclient import version as swiftclient_version
from swiftclient.exceptions import ClientException
from swiftclient.requests_compat import SwiftClientRequestsSession
from swiftclient.utils import (
def get_auth_keystone(auth_url, user, key, os_options, **kwargs):
    """
    Authenticate against a keystone server.

    We are using the keystoneclient library for authentication.
    """
    insecure = kwargs.get('insecure', False)
    timeout = kwargs.get('timeout', None)
    auth_version = kwargs.get('auth_version', None)
    debug = logger.isEnabledFor(logging.DEBUG)
    if not VERSIONFUL_AUTH_PATH.match(urlparse(auth_url).path.rstrip('/').rsplit('/', 1)[-1]):
        auth_url = auth_url.rstrip('/') + '/'
        if auth_version and auth_version in AUTH_VERSIONS_V2:
            auth_url = urljoin(auth_url, 'v2.0')
        else:
            auth_url = urljoin(auth_url, 'v3')
            auth_version = '3'
        logger.debug('Versionless auth_url - using %s as endpoint' % auth_url)
    if auth_version is None:
        auth_version = '2'
    ksclient = None
    if auth_version in AUTH_VERSIONS_V3:
        if ksclient_v3 is not None:
            ksclient = ksclient_v3
    elif ksclient_v2 is not None:
        ksclient = ksclient_v2
    if ksclient is None:
        raise ClientException('\nAuth versions 2.0 and 3 require python-keystoneclient, install it or use Auth\nversion 1.0 which requires ST_AUTH, ST_USER, and ST_KEY environment\nvariables to be set or overridden with -A, -U, or -K.')
    filter_kwargs = {}
    service_type = os_options.get('service_type') or 'object-store'
    endpoint_type = os_options.get('endpoint_type') or 'publicURL'
    if os_options.get('region_name'):
        filter_kwargs['attr'] = 'region'
        filter_kwargs['filter_value'] = os_options['region_name']
    if os_options.get('auth_type') and os_options['auth_type'] not in ('password', 'v2password', 'v3password', 'v3applicationcredential'):
        raise ClientException('Swiftclient currently only supports v3applicationcredential for auth_type')
    elif os_options.get('auth_type') == 'v3applicationcredential':
        if ksa_v3 is None:
            raise ClientException('Auth v3applicationcredential requires keystoneauth1 package; consider upgrading to python-keystoneclient>=2.0.0')
        try:
            auth = ksa_v3.ApplicationCredential(auth_url=auth_url, application_credential_secret=os_options.get('application_credential_secret'), application_credential_id=os_options.get('application_credential_id'))
            sess = ksa_session.Session(auth=auth)
            token = sess.get_token()
        except ksauthexceptions.Unauthorized:
            msg = 'Unauthorized. Check application credential id and secret.'
            raise ClientException(msg)
        except ksauthexceptions.AuthorizationFailure as err:
            raise ClientException('Authorization Failure. %s' % err)
        try:
            endpoint = sess.get_endpoint_data(service_type=service_type, endpoint_type=endpoint_type, **filter_kwargs)
            return (endpoint.catalog_url, token)
        except ksauthexceptions.EndpointNotFound:
            raise ClientException('Endpoint for %s not found - have you specified a region?' % service_type)
    try:
        _ksclient = ksclient.Client(username=user, password=key, token=os_options.get('auth_token'), tenant_name=os_options.get('tenant_name'), tenant_id=os_options.get('tenant_id'), user_id=os_options.get('user_id'), user_domain_name=os_options.get('user_domain_name'), user_domain_id=os_options.get('user_domain_id'), project_name=os_options.get('project_name'), project_id=os_options.get('project_id'), project_domain_name=os_options.get('project_domain_name'), project_domain_id=os_options.get('project_domain_id'), debug=debug, cacert=kwargs.get('cacert'), cert=kwargs.get('cert'), key=kwargs.get('cert_key'), auth_url=auth_url, insecure=insecure, timeout=timeout)
    except ksexceptions.Unauthorized:
        msg = 'Unauthorized. Check username, password and tenant name/id.'
        if auth_version in AUTH_VERSIONS_V3:
            msg = 'Unauthorized. Check username/id, password, tenant name/id and user/tenant domain name/id.'
        raise ClientException(msg)
    except ksexceptions.AuthorizationFailure as err:
        raise ClientException('Authorization Failure. %s' % err)
    try:
        endpoint = _ksclient.service_catalog.url_for(service_type=service_type, endpoint_type=endpoint_type, **filter_kwargs)
    except ksexceptions.EndpointNotFound:
        raise ClientException('Endpoint for %s not found - have you specified a region?' % service_type)
    return (endpoint, _ksclient.auth_token)