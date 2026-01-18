import io
import tempfile
import textwrap
import six
from six.moves import configparser
import sys
from pbr.tests import base
from pbr import util
def test_extras_parsing(self):
    config = config_from_ini(self.config_text)
    kwargs = util.setup_cfg_to_setup_kwargs(config)
    self.assertEqual(self.expected_extra_requires, kwargs['extras_require'])