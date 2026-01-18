import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
@property
def secret_refs(self):
    if self._cached_secrets:
        self._secret_refs = dict(((name, secret.secret_ref) for name, secret in self._cached_secrets.items()))
    return self._secret_refs