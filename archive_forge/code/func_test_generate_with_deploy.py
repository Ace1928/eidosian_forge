import os
import textwrap
import unittest
from gae_ext_runtime import testutil
def test_generate_with_deploy(self):
    self.write_file('index.php', 'index')
    self.generate_configs(deploy=True)
    dockerfile = self.file_contents('Dockerfile')
    self.assertEqual(dockerfile, textwrap.dedent('            # Dockerfile extending the generic PHP image with application files for a\n            # single application.\n            FROM gcr.io/google-appengine/php:latest\n\n            # The Docker image will configure the document root according to this\n            # environment variable.\n            ENV DOCUMENT_ROOT /app\n            '))
    dockerignore = self.file_contents('.dockerignore')
    self.assertEqual(dockerignore, self.license() + textwrap.dedent('            .dockerignore\n            Dockerfile\n            .git\n            .hg\n            .svn\n            '))