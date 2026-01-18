import glanceclient
from keystoneauth1 import loading
from keystoneauth1 import session
import os
import os_client_config
from tempest.lib.cli import base
def glance_pyclient(self):
    ks_creds = dict(auth_url=self.creds['auth_url'], username=self.creds['username'], password=self.creds['password'], project_name=self.creds['project_name'], user_domain_id=self.creds['user_domain_id'], project_domain_id=self.creds['project_domain_id'])
    keystoneclient = self.Keystone(**ks_creds)
    return self.Glance(keystoneclient)