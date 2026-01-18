import os
import textwrap
import unittest
from gae_ext_runtime import testutil
def test_generate_custom_runtime(self):
    self.write_file('index.php', 'index')
    self.generate_configs(custom=True)
    dockerfile = self.file_contents('Dockerfile')
    self.assertEqual(dockerfile, self.preamble() + textwrap.dedent('            ENV DOCUMENT_ROOT /app\n            '))
    self.assert_file_exists_with_contents('.dockerignore', self.license() + textwrap.dedent('            .dockerignore\n            Dockerfile\n            .git\n            .hg\n            .svn\n            '))