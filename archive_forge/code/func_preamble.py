import os
import textwrap
import unittest
from gae_ext_runtime import testutil
def preamble(self):
    return textwrap.dedent('            # Dockerfile extending the generic PHP image with application files for a\n            # single application.\n            FROM gcr.io/google-appengine/php:latest\n\n            # The Docker image will configure the document root according to this\n            # environment variable.\n            ')