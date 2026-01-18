import os
import tempfile
import textwrap
from openstack.config import loader
from openstack import exceptions
from openstack.tests.unit.config import base
def test__load_yaml_json_file_without_perm(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        tested_files = []
        fn = os.path.join(tmpdir, 'file.txt')
        with open(fn, 'w+') as fp:
            fp.write(FILES['txt'])
        os.chmod(fn, 222)
        tested_files.append(fn)
        path, result = loader.OpenStackConfig()._load_yaml_json_file(tested_files)
        self.assertEqual(None, path)