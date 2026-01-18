import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
@private_key_passphrase.setter
@_immutable_after_save
def private_key_passphrase(self, value):
    super(CertificateContainer, self).remove('private_key_passphrase')
    super(CertificateContainer, self).add('private_key_passphrase', value)