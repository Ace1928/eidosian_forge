import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
@property
def is_valid_with_password(self):
    return self.host is not None and self.username is not None and (self.password is not None)