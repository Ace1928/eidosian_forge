from pprint import pformat
from six import iteritems
import re
@attacher.setter
def attacher(self, attacher):
    """
        Sets the attacher of this V1beta1VolumeAttachmentSpec.
        Attacher indicates the name of the volume driver that MUST handle this
        request. This is the name returned by GetPluginName().

        :param attacher: The attacher of this V1beta1VolumeAttachmentSpec.
        :type: str
        """
    if attacher is None:
        raise ValueError('Invalid value for `attacher`, must not be `None`')
    self._attacher = attacher