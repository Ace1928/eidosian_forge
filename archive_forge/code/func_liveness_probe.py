from pprint import pformat
from six import iteritems
import re
@liveness_probe.setter
def liveness_probe(self, liveness_probe):
    """
        Sets the liveness_probe of this V1Container.
        Periodic probe of container liveness. Container will be restarted if the
        probe fails. Cannot be updated. More info:
        https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle#container-probes

        :param liveness_probe: The liveness_probe of this V1Container.
        :type: V1Probe
        """
    self._liveness_probe = liveness_probe