from pprint import pformat
from six import iteritems
import re
@endpoints_namespace.setter
def endpoints_namespace(self, endpoints_namespace):
    """
        Sets the endpoints_namespace of this V1GlusterfsPersistentVolumeSource.
        EndpointsNamespace is the namespace that contains Glusterfs endpoint. If
        this field is empty, the EndpointNamespace defaults to the same
        namespace as the bound PVC. More info:
        https://releases.k8s.io/HEAD/examples/volumes/glusterfs/README.md#create-a-pod

        :param endpoints_namespace: The endpoints_namespace of this
        V1GlusterfsPersistentVolumeSource.
        :type: str
        """
    self._endpoints_namespace = endpoints_namespace