from yowsup.config.base import config
import logging
@password.setter
def password(self, value):
    self._password = value
    if value is not None:
        logger.warn('Setting a password in Config is deprecated and not used anymore. client_static_keypair is used instead')