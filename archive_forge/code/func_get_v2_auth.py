import argparse
import collections
import getpass
import logging
import sys
from urllib import parse as urlparse
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import loading
from keystoneauth1 import session
from oslo_utils import importutils
import requests
import cinderclient
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import client
from cinderclient import exceptions as exc
from cinderclient import utils
def get_v2_auth(self, v2_auth_url):
    username = self.options.os_username
    password = self.options.os_password
    tenant_id = self.options.os_project_id
    tenant_name = self.options.os_project_name
    return v2_auth.Password(v2_auth_url, username=username, password=password, tenant_id=tenant_id, tenant_name=tenant_name)