import unittest
import time
from nose.plugins.attrib import attr
from boto.redshift.layer1 import RedshiftConnection
from boto.redshift.exceptions import ClusterNotFoundFault
from boto.redshift.exceptions import ResizeNotFoundFault
@attr('notdefault')
def test_create_delete_cluster(self):
    cluster_id = self.cluster_id()
    self.api.create_cluster(cluster_id, self.node_type, self.master_username, self.master_password, db_name=self.db_name, number_of_nodes=3)
    time.sleep(self.wait_time)
    self.api.delete_cluster(cluster_id, skip_final_cluster_snapshot=True)