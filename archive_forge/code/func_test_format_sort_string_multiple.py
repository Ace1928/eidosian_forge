from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_format_sort_string_multiple(self):
    s = 'id:asc,status,size:desc'
    self.assertEqual('id:asc,status,size:desc', cs.volumes._format_sort_param(s))