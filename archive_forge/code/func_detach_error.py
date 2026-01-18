from pprint import pformat
from six import iteritems
import re
@detach_error.setter
def detach_error(self, detach_error):
    """
        Sets the detach_error of this V1alpha1VolumeAttachmentStatus.
        The last error encountered during detach operation, if any. This field
        must only be set by the entity completing the detach operation, i.e. the
        external-attacher.

        :param detach_error: The detach_error of this
        V1alpha1VolumeAttachmentStatus.
        :type: V1alpha1VolumeError
        """
    self._detach_error = detach_error