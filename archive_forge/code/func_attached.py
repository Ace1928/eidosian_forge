from pprint import pformat
from six import iteritems
import re
@attached.setter
def attached(self, attached):
    """
        Sets the attached of this V1alpha1VolumeAttachmentStatus.
        Indicates the volume is successfully attached. This field must only be
        set by the entity completing the attach operation, i.e. the
        external-attacher.

        :param attached: The attached of this V1alpha1VolumeAttachmentStatus.
        :type: bool
        """
    if attached is None:
        raise ValueError('Invalid value for `attached`, must not be `None`')
    self._attached = attached