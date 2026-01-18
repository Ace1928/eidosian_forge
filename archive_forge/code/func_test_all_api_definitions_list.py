import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def test_all_api_definitions_list(self):
    ext_aliases = []
    api_def_path = 'neutron_lib/api/definitions'
    for f in sorted(os.listdir(api_def_path)):
        mod_name, file_ext = os.path.splitext(os.path.split(f)[-1])
        ext_path = os.path.join(api_def_path, f)
        if file_ext.lower() == '.py' and (not mod_name.startswith('_')):
            mod = self._load_module(mod_name, ext_path)
            ext_alias = getattr(mod, 'ALIAS', None)
            if not ext_alias:
                continue
            ext_aliases.append(ext_alias)
    self.assertEqual(sorted(ext_aliases), sorted([d.ALIAS for d in definitions._ALL_API_DEFINITIONS]))