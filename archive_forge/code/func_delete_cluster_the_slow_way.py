import unittest
import time
from nose.plugins.attrib import attr
from boto.redshift.layer1 import RedshiftConnection
from boto.redshift.exceptions import ClusterNotFoundFault
from boto.redshift.exceptions import ResizeNotFoundFault
def delete_cluster_the_slow_way(self, cluster_id):
    time.sleep(self.wait_time)
    self.api.delete_cluster(cluster_id, skip_final_cluster_snapshot=True)