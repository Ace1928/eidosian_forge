from pprint import pformat
from six import iteritems
import re
@attach_required.setter
def attach_required(self, attach_required):
    """
        Sets the attach_required of this V1beta1CSIDriverSpec.
        attachRequired indicates this CSI volume driver requires an attach
        operation (because it implements the CSI ControllerPublishVolume()
        method), and that the Kubernetes attach detach controller should call
        the attach volume interface which checks the volumeattachment status and
        waits until the volume is attached before proceeding to mounting. The
        CSI external-attacher coordinates with CSI volume driver and updates the
        volumeattachment status when the attach operation is complete. If the
        CSIDriverRegistry feature gate is enabled and the value is specified to
        false, the attach operation will be skipped. Otherwise the attach
        operation will be called.

        :param attach_required: The attach_required of this
        V1beta1CSIDriverSpec.
        :type: bool
        """
    self._attach_required = attach_required