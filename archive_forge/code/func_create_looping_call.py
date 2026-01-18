from os_ken.lib import hub
import logging
@staticmethod
def create_looping_call(funct, *args, **kwargs):
    LOG.debug('create_looping_call called')
    return LoopingCall(funct, *args, **kwargs)