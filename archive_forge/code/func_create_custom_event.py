from os_ken.lib import hub
import logging
@staticmethod
def create_custom_event():
    LOG.debug('Create CustomEvent called')
    return hub.Event()