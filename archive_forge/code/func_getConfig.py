import boto
import boto.utils
from boto.compat import StringIO
from boto.mashups.iobject import IObject
from boto.pyami.config import Config, BotoConfigPath
from boto.mashups.interactive import interactive_shell
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty
import os
def getConfig(self):
    if not self._config:
        remote_file = BotoConfigPath
        local_file = '%s.ini' % self.instance.id
        self.get_file(remote_file, local_file)
        self._config = Config(local_file)
    return self._config