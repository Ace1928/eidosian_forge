import os
from urllib import parse
import tempest.lib.cli.base
from novaclient import client
from novaclient.tests.functional import base
def nova_auth_with_password(self, action, identity_api_version):
    flags = f'--os-username {self.cli_clients.username} --os-tenant-name {self.cli_clients.tenant_name} --os-password {self.cli_clients.password} --os-auth-url {self._get_url(identity_api_version)} --os-endpoint-type publicURL'
    if self.cacert:
        flags = f'{flags} --os-cacert {self.cacert}'
    if self.cert:
        flags = f'{flags} --os-cert {self.cert}'
    if self.cli_clients.insecure:
        flags = f'{flags} --insecure'
    return tempest.lib.cli.base.execute('nova', action, flags, cli_dir=self.cli_clients.cli_dir)