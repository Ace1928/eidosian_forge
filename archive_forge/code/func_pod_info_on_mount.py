from pprint import pformat
from six import iteritems
import re
@pod_info_on_mount.setter
def pod_info_on_mount(self, pod_info_on_mount):
    """
        Sets the pod_info_on_mount of this V1beta1CSIDriverSpec.
        If set to true, podInfoOnMount indicates this CSI volume driver requires
        additional pod information (like podName, podUID, etc.) during mount
        operations. If set to false, pod information will not be passed on
        mount. Default is false. The CSI driver specifies podInfoOnMount as part
        of driver deployment. If true, Kubelet will pass pod information as
        VolumeContext in the CSI NodePublishVolume() calls. The CSI driver is
        responsible for parsing and validating the information passed in as
        VolumeContext. The following VolumeConext will be passed if
        podInfoOnMount is set to true. This list might grow, but the prefix will
        be used. "csi.storage.k8s.io/pod.name": pod.Name
        "csi.storage.k8s.io/pod.namespace": pod.Namespace
        "csi.storage.k8s.io/pod.uid": string(pod.UID)

        :param pod_info_on_mount: The pod_info_on_mount of this
        V1beta1CSIDriverSpec.
        :type: bool
        """
    self._pod_info_on_mount = pod_info_on_mount