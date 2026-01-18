from pprint import pformat
from six import iteritems
import re
@involved_object.setter
def involved_object(self, involved_object):
    """
        Sets the involved_object of this V1Event.
        The object that this event is about.

        :param involved_object: The involved_object of this V1Event.
        :type: V1ObjectReference
        """
    if involved_object is None:
        raise ValueError('Invalid value for `involved_object`, must not be `None`')
    self._involved_object = involved_object