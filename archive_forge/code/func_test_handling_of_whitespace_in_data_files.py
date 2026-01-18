import io
import tempfile
import textwrap
import six
from six.moves import configparser
import sys
from pbr.tests import base
from pbr import util
def test_handling_of_whitespace_in_data_files(self):
    config = config_from_ini(self.config_text)
    kwargs = util.setup_cfg_to_setup_kwargs(config)
    self.assertEqual(self.data_files, kwargs['data_files'])