import os
import os_client_config
from tempest.lib.cli import base
def mistral(self, action, flags='', params='', fail_ok=False):
    """Executes Mistral command."""
    mistral_url_op = '--os-mistral-url %s' % self._mistral_url
    flags = '{} --insecure'.format(flags)
    if 'WITHOUT_AUTH' in os.environ:
        return base.execute('mistral %s' % mistral_url_op, action, flags, params, fail_ok, merge_stderr=False, cli_dir='')
    else:
        return self.clients.cmd_with_auth('mistral %s' % mistral_url_op, action, flags, params, fail_ok)