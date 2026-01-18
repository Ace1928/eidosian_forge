from __future__ import absolute_import
import urllib3
import copy
import logging
import multiprocessing
import sys
from six import iteritems
from six import with_metaclass
from six.moves import http_client as httplib
def get_api_key_with_prefix(self, identifier):
    """
        Gets API key (with prefix if set).

        :param identifier: The identifier of apiKey.
        :return: The token for api key authentication.
        """
    if self.api_key.get(identifier) and self.api_key_prefix.get(identifier):
        return self.api_key_prefix[identifier] + ' ' + self.api_key[identifier]
    elif self.api_key.get(identifier):
        return self.api_key[identifier]