import abc
import logging
from six import with_metaclass
@classmethod
def getCurrent(cls):
    """
        :rtype: YowsupEnv
        """
    if cls.__CURR is None:
        env = DEFAULT
        envs = cls.getRegisteredEnvs()
        if env not in envs:
            env = envs[0]
        logger.debug('Env not set, setting it to %s' % env)
        cls.setEnv(env)
    return cls.__CURR