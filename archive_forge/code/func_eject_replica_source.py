import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def eject_replica_source(self, instance):
    """Eject a replica source from its set

        :param instance: The :class:`Instance` (or its ID) of the database
                         instance to eject.
        """
    body = {'eject_replica_source': {}}
    self._action(instance, body)