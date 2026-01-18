from __future__ import absolute_import
import urllib3
import copy
import logging
import multiprocessing
import sys
from six import iteritems
from six import with_metaclass
from six.moves import http_client as httplib
@logger_format.setter
def logger_format(self, value):
    """
        Sets the logger_format.

        The logger_formatter will be updated when sets logger_format.

        :param value: The format string.
        :type: str
        """
    self.__logger_format = value
    self.logger_formatter = logging.Formatter(self.__logger_format)