from pprint import pformat
from six import iteritems
import re
@dataset_uuid.setter
def dataset_uuid(self, dataset_uuid):
    """
        Sets the dataset_uuid of this V1FlockerVolumeSource.
        UUID of the dataset. This is unique identifier of a Flocker dataset

        :param dataset_uuid: The dataset_uuid of this V1FlockerVolumeSource.
        :type: str
        """
    self._dataset_uuid = dataset_uuid