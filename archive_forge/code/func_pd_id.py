from pprint import pformat
from six import iteritems
import re
@pd_id.setter
def pd_id(self, pd_id):
    """
        Sets the pd_id of this V1PhotonPersistentDiskVolumeSource.
        ID that identifies Photon Controller persistent disk

        :param pd_id: The pd_id of this V1PhotonPersistentDiskVolumeSource.
        :type: str
        """
    if pd_id is None:
        raise ValueError('Invalid value for `pd_id`, must not be `None`')
    self._pd_id = pd_id