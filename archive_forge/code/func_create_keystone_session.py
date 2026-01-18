from collections import namedtuple
import logging
import sys
from cliff import app
from cliff import command
from cliff import commandmanager
from cliff import complete
from cliff import help
from keystoneauth1 import identity
from keystoneauth1 import loading
from keystoneauth1 import session
import barbicanclient
from barbicanclient._i18n import _LW
from barbicanclient import client
def create_keystone_session(self, args, api_version, kwargs_dict, auth_type):
    self.check_auth_arguments(args, api_version, raise_exc=True)
    kwargs = self.build_kwargs_based_on_version(args, api_version)
    kwargs.update(kwargs_dict)
    _supported_version = _IDENTITY_API_VERSION_2 + _IDENTITY_API_VERSION_3
    if not api_version or api_version not in _supported_version:
        self.stderr.write('WARNING: The identity version <{0}> is not in supported versions <{1}>, falling back to <{2}>.'.format(api_version, _IDENTITY_API_VERSION_2 + _IDENTITY_API_VERSION_3, _DEFAULT_IDENTITY_API_VERSION))
    method = identity.Token if auth_type == 'token' else identity.Password
    auth = method(**kwargs)
    cacert = args.os_cacert
    cert = args.os_cert
    key = args.os_key
    insecure = args.insecure
    if insecure:
        verify = False
    else:
        verify = cacert or True
    if cert and key:
        cert = (cert, key)
    return session.Session(auth=auth, verify=verify, cert=cert)