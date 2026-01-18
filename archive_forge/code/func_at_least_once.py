import logging
from debtcollector import removals
from oslo_config import cfg
from stevedore import driver
from urllib import parse
from oslo_messaging import exceptions
@property
def at_least_once(self):
    return self._at_least_once