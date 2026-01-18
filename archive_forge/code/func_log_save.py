import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def log_save(self, instance, log_name, filename=None):
    """Saves a guest log to a file.

        :param instance: The :class:`Instance` (or its ID) of the database
                         instance to get the log for.
        :param log_name: The name of <log> to publish
        :rtype: Filename to which log was saved
        """
    written_file = filename or 'trove-' + instance.id + '-' + log_name + '.log'
    log_gen = self.log_generator(instance, log_name, lines=0)
    with open(written_file, 'w') as f:
        for log_obj in log_gen():
            f.write(log_obj)
    return written_file