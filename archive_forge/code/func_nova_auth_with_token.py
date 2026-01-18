import os
from urllib import parse
import tempest.lib.cli.base
from novaclient import client
from novaclient.tests.functional import base
def nova_auth_with_token(self, identity_api_version):
    auth_ref = self.client.client.session.auth.get_access(self.client.client.session)
    token = auth_ref.auth_token
    auth_url = self._get_url(identity_api_version)
    kw = {}
    if identity_api_version == '3':
        kw['project_domain_id'] = self.project_domain_id
    nova = client.Client('2', auth_token=token, auth_url=auth_url, project_name=self.project_name, cacert=self.cacert, cert=self.cert, **kw)
    nova.servers.list()
    os.environ.pop('OS_AUTH_TYPE', None)
    os.environ.pop('OS_AUTH_PLUGIN', None)
    flags = f'--os-tenant-name {self.project_name} --os-token {token} --os-auth-url {auth_url} --os-endpoint-type publicURL'
    if self.cacert:
        flags = f'{flags} --os-cacert {self.cacert}'
    if self.cert:
        flags = f'{flags} --os-cert {self.cert}'
    if self.cli_clients.insecure:
        flags = f'{flags} --insecure'
    tempest.lib.cli.base.execute('nova', 'list', flags, cli_dir=self.cli_clients.cli_dir)