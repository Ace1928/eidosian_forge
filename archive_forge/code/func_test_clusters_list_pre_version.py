import ddt
from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@ddt.data(True, False)
def test_clusters_list_pre_version(self, detailed):
    pre_cs = fakes.FakeClient(api_version=api_versions.APIVersion('3.6'))
    self.assertRaises(exc.VersionNotFoundForAPIMethod, pre_cs.clusters.list, detailed=detailed)