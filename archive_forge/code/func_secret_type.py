import base64
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
@secret_type.setter
@immutable_after_save
def secret_type(self, value):
    self._secret_type = value