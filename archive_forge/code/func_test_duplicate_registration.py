import io
import tempfile
import textwrap
from oslotest import base
from oslo_config import cfg
def test_duplicate_registration(self):
    dupe_opt = cfg.StrOpt('normal_opt', default='normal_opt_default')
    dupe_opt._set_location = cfg.LocationInfo(cfg.Locations.opt_default, 'an alternative file')
    self.assertFalse(self.conf.register_opt(dupe_opt))