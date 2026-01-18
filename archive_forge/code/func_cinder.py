from pprint import pformat
from six import iteritems
import re
@cinder.setter
def cinder(self, cinder):
    """
        Sets the cinder of this V1PersistentVolumeSpec.
        Cinder represents a cinder volume attached and mounted on kubelets host
        machine More info:
        https://releases.k8s.io/HEAD/examples/mysql-cinder-pd/README.md

        :param cinder: The cinder of this V1PersistentVolumeSpec.
        :type: V1CinderPersistentVolumeSource
        """
    self._cinder = cinder