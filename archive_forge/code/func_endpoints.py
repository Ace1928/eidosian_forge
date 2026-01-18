from pprint import pformat
from six import iteritems
import re
@endpoints.setter
def endpoints(self, endpoints):
    """
        Sets the endpoints of this V1GlusterfsVolumeSource.
        EndpointsName is the endpoint name that details Glusterfs topology. More
        info:
        https://releases.k8s.io/HEAD/examples/volumes/glusterfs/README.md#create-a-pod

        :param endpoints: The endpoints of this V1GlusterfsVolumeSource.
        :type: str
        """
    if endpoints is None:
        raise ValueError('Invalid value for `endpoints`, must not be `None`')
    self._endpoints = endpoints