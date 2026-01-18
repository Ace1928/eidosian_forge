import abc
import logging
from six import with_metaclass
@classmethod
def setEnv(cls, envName):
    if not envName in cls.__ENVS:
        raise ValueError('%s env does not exist' % envName)
    logger.debug('Current env changed to %s ' % envName)
    cls.__CURR = cls.__ENVS[envName]()