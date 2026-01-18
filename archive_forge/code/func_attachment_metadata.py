from pprint import pformat
from six import iteritems
import re
@attachment_metadata.setter
def attachment_metadata(self, attachment_metadata):
    """
        Sets the attachment_metadata of this V1alpha1VolumeAttachmentStatus.
        Upon successful attach, this field is populated with any information
        returned by the attach operation that must be passed into subsequent
        WaitForAttach or Mount calls. This field must only be set by the entity
        completing the attach operation, i.e. the external-attacher.

        :param attachment_metadata: The attachment_metadata of this
        V1alpha1VolumeAttachmentStatus.
        :type: dict(str, str)
        """
    self._attachment_metadata = attachment_metadata