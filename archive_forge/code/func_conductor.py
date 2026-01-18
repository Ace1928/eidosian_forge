import abc
import os
import threading
import fasteners
from taskflow import engines
from taskflow import exceptions as excp
from taskflow.types import entity
from taskflow.types import notifier
from taskflow.utils import misc
@misc.cachedproperty
def conductor(self):
    """Entity object that represents this conductor."""
    hostname = misc.get_hostname()
    pid = os.getpid()
    name = '@'.join([self._name, hostname + ':' + str(pid)])
    metadata = {'hostname': hostname, 'pid': pid}
    return entity.Entity(self.ENTITY_KIND, name, metadata)