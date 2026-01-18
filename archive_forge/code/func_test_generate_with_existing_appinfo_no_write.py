import os
import textwrap
import unittest
from gae_ext_runtime import testutil
def test_generate_with_existing_appinfo_no_write(self):
    """Tests generate_config_data with fake appinfo."""
    self.write_file('index.php', 'index')
    appinfo = testutil.AppInfoFake(runtime_config={'document_root': 'wordpress'}, entrypoint='["/bin/bash", "my-cmd.sh"]')
    cfg_files = self.generate_config_data(deploy=True, appinfo=appinfo)
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', self.preamble() + textwrap.dedent('            ENV DOCUMENT_ROOT /app/wordpress\n\n            # Allow custom CMD\n            CMD ["/bin/bash", "my-cmd.sh"]\n            '))
    self.assert_genfile_exists_with_contents(cfg_files, '.dockerignore', self.license() + textwrap.dedent('            .dockerignore\n            Dockerfile\n            .git\n            .hg\n            .svn\n            '))