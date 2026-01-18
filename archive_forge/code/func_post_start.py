from pprint import pformat
from six import iteritems
import re
@post_start.setter
def post_start(self, post_start):
    """
        Sets the post_start of this V1Lifecycle.
        PostStart is called immediately after a container is created. If the
        handler fails, the container is terminated and restarted according to
        its restart policy. Other management of the container blocks until the
        hook completes. More info:
        https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/#container-hooks

        :param post_start: The post_start of this V1Lifecycle.
        :type: V1Handler
        """
    self._post_start = post_start