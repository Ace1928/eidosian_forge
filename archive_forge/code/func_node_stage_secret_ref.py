from pprint import pformat
from six import iteritems
import re
@node_stage_secret_ref.setter
def node_stage_secret_ref(self, node_stage_secret_ref):
    """
        Sets the node_stage_secret_ref of this V1CSIPersistentVolumeSource.
        NodeStageSecretRef is a reference to the secret object containing
        sensitive information to pass to the CSI driver to complete the CSI
        NodeStageVolume and NodeStageVolume and NodeUnstageVolume calls. This
        field is optional, and may be empty if no secret is required. If the
        secret object contains more than one secret, all secrets are passed.

        :param node_stage_secret_ref: The node_stage_secret_ref of this
        V1CSIPersistentVolumeSource.
        :type: V1SecretReference
        """
    self._node_stage_secret_ref = node_stage_secret_ref