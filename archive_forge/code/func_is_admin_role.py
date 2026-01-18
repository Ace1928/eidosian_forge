import abc
import argparse
import functools
import logging
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def is_admin_role(self):
    client = self.get_client()
    auth_ref = client.httpclient.get_auth_ref()
    if not auth_ref:
        return False
    return 'admin' in auth_ref.role_names