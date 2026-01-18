from pprint import pformat
from six import iteritems
import re
@proc_mount.setter
def proc_mount(self, proc_mount):
    """
        Sets the proc_mount of this V1SecurityContext.
        procMount denotes the type of proc mount to use for the containers. The
        default is DefaultProcMount which uses the container runtime defaults
        for readonly paths and masked paths. This requires the ProcMountType
        feature flag to be enabled.

        :param proc_mount: The proc_mount of this V1SecurityContext.
        :type: str
        """
    self._proc_mount = proc_mount