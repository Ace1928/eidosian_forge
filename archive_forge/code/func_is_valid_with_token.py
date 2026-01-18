import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
@property
def is_valid_with_token(self):
    return self.host is not None and self.token is not None