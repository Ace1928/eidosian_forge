from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
def test_cluster_create_success_only_clustertemplate_arg(self):
    self._test_cluster_create_success('cluster-create --cluster-template xxx', ['xxx'], {})