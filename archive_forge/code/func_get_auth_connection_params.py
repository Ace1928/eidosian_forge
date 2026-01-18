import datetime
import functools
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import urllib
import uuid
import requests
import keystoneauth1
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
def get_auth_connection_params(self, auth=None, **kwargs):
    """Return auth connection params as provided by the auth plugin.

        An auth plugin may specify connection parameters to the request like
        providing a client certificate for communication.

        We restrict the values that may be returned from this function to
        prevent an auth plugin overriding values unrelated to connection
        parmeters. The values that are currently accepted are:

        - `cert`: a path to a client certificate, or tuple of client
          certificate and key pair that are used with this request.
        - `verify`: a boolean value to indicate verifying SSL certificates
          against the system CAs or a path to a CA file to verify with.

        These values are passed to the requests library and further information
        on accepted values may be found there.

        :param auth: The auth plugin to use for tokens. Overrides the plugin
                     on the session. (optional)
        :type auth: keystoneauth1.plugin.BaseAuthPlugin

        :raises keystoneauth1.exceptions.auth.AuthorizationFailure:
            if a new token fetch fails.
        :raises keystoneauth1.exceptions.auth_plugins.MissingAuthPlugin:
            if a plugin is not available.
        :raises keystoneauth1.exceptions.auth_plugins.UnsupportedParameters:
            if the plugin returns a parameter that is not supported by this
            session.

        :returns: Authentication headers or None for failure.
        :rtype: :class:`dict`
        """
    auth = self._auth_required(auth, 'fetch connection params')
    params = auth.get_connection_params(self, **kwargs)
    params_copy = params.copy()
    for arg in ('cert', 'verify'):
        try:
            kwargs[arg] = params_copy.pop(arg)
        except KeyError:
            pass
    if params_copy:
        raise exceptions.UnsupportedParameters(list(params_copy.keys()))
    return params