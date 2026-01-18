from pprint import pformat
from six import iteritems
import re
@initializers.setter
def initializers(self, initializers):
    """
        Sets the initializers of this V1ObjectMeta.
        An initializer is a controller which enforces some system invariant at
        object creation time. This field is a list of initializers that have not
        yet acted on this object. If nil or empty, this object has been
        completely initialized. Otherwise, the object is considered
        uninitialized and is hidden (in list/watch and get calls) from clients
        that haven't explicitly asked to observe uninitialized objects.  When an
        object is created, the system will populate this list with the current
        set of initializers. Only privileged users may set or modify this list.
        Once it is empty, it may not be modified further by any user.
        DEPRECATED - initializers are an alpha field and will be removed in
        v1.15.

        :param initializers: The initializers of this V1ObjectMeta.
        :type: V1Initializers
        """
    self._initializers = initializers