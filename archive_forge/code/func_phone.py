from yowsup.config.base import config
import logging
@phone.setter
def phone(self, value):
    self._phone = str(value) if value is not None else None