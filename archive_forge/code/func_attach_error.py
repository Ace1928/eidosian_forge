from pprint import pformat
from six import iteritems
import re
@attach_error.setter
def attach_error(self, attach_error):
    """
        Sets the attach_error of this V1alpha1VolumeAttachmentStatus.
        The last error encountered during attach operation, if any. This field
        must only be set by the entity completing the attach operation, i.e. the
        external-attacher.

        :param attach_error: The attach_error of this
        V1alpha1VolumeAttachmentStatus.
        :type: V1alpha1VolumeError
        """
    self._attach_error = attach_error