import abc
import logging
from six import with_metaclass
@classmethod
def getRegisteredEnvs(cls):
    return list(cls.__ENVS.keys())