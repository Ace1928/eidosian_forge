from pprint import pformat
from six import iteritems
import re
@reclaim_policy.setter
def reclaim_policy(self, reclaim_policy):
    """
        Sets the reclaim_policy of this V1beta1StorageClass.
        Dynamically provisioned PersistentVolumes of this storage class are
        created with this reclaimPolicy. Defaults to Delete.

        :param reclaim_policy: The reclaim_policy of this V1beta1StorageClass.
        :type: str
        """
    self._reclaim_policy = reclaim_policy