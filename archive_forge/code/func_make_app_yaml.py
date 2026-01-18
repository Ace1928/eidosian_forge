import logging
import mock
import os
import re
import sys
import shutil
import tempfile
import textwrap
import unittest
from gae_ext_runtime import testutil
from gae_ext_runtime import ext_runtime
import constants
def make_app_yaml(self, runtime):
    return 'env: flex\nruntime: {runtime}\n'.format(runtime=runtime)