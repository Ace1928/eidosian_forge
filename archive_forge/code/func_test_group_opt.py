import io
import tempfile
import textwrap
from oslotest import base
from oslo_config import cfg
def test_group_opt(self):
    self.conf([])
    loc = self.conf.get_location('group_opt', 'group')
    self.assertEqual(cfg.Locations.opt_default, loc.location)
    self.assertIn('test_get_location.py', loc.detail)