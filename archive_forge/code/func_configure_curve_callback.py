import logging
import os
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
import zmq
from zmq.error import _check_version
from zmq.utils import z85
from .certs import load_certificates
def configure_curve_callback(self, domain: str='*', credentials_provider: Any=None) -> None:
    """Configure CURVE authentication for a given domain.

        CURVE authentication using a callback function validating
        the client public key according to a custom mechanism, e.g. checking the
        key against records in a db. credentials_provider is an object of a class which
        implements a callback method accepting two parameters (domain and key), e.g.::

            class CredentialsProvider(object):

                def __init__(self):
                    ...e.g. db connection

                def callback(self, domain, key):
                    valid = ...lookup key and/or domain in db
                    if valid:
                        logging.info('Authorizing: {0}, {1}'.format(domain, key))
                        return True
                    else:
                        logging.warning('NOT Authorizing: {0}, {1}'.format(domain, key))
                        return False

        To cover all domains, use "*".
        """
    self.allow_any = False
    if credentials_provider is not None:
        self.credentials_providers[domain] = credentials_provider
    else:
        self.log.error('None credentials_provider provided for domain:%s', domain)