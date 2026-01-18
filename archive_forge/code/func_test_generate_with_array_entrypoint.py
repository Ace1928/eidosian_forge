import os
import textwrap
import unittest
from gae_ext_runtime import testutil
def test_generate_with_array_entrypoint(self):
    self.write_file('index.php', 'index')
    appinfo = testutil.AppInfoFake(runtime_config={'document_root': 'wordpress'}, entrypoint=['/bin/bash', 'my-cmd.sh'])
    self.generate_configs(deploy=True, appinfo=appinfo)
    dockerfile = self.file_contents('Dockerfile')
    self.assertEqual(dockerfile, self.preamble() + textwrap.dedent('            ENV DOCUMENT_ROOT /app/wordpress\n\n            # Allow custom CMD\n            CMD ["/bin/bash", "my-cmd.sh"]\n            '))