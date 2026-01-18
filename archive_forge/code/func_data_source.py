from pprint import pformat
from six import iteritems
import re
@data_source.setter
def data_source(self, data_source):
    """
        Sets the data_source of this V1PersistentVolumeClaimSpec.
        This field requires the VolumeSnapshotDataSource alpha feature gate to
        be enabled and currently VolumeSnapshot is the only supported data
        source. If the provisioner can support VolumeSnapshot data source, it
        will create a new volume and data will be restored to the volume at the
        same time. If the provisioner does not support VolumeSnapshot data
        source, volume will not be created and the failure will be reported as
        an event. In the future, we plan to support more data source types and
        the behavior of the provisioner may change.

        :param data_source: The data_source of this V1PersistentVolumeClaimSpec.
        :type: V1TypedLocalObjectReference
        """
    self._data_source = data_source