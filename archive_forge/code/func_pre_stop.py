from pprint import pformat
from six import iteritems
import re
@pre_stop.setter
def pre_stop(self, pre_stop):
    """
        Sets the pre_stop of this V1Lifecycle.
        PreStop is called immediately before a container is terminated due to an
        API request or management event such as liveness probe failure,
        preemption, resource contention, etc. The handler is not called if the
        container crashes or exits. The reason for termination is passed to the
        handler. The Pod's termination grace period countdown begins before the
        PreStop hooked is executed. Regardless of the outcome of the handler,
        the container will eventually terminate within the Pod's termination
        grace period. Other management of the container blocks until the hook
        completes or until the termination grace period is reached. More info:
        https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/#container-hooks

        :param pre_stop: The pre_stop of this V1Lifecycle.
        :type: V1Handler
        """
    self._pre_stop = pre_stop