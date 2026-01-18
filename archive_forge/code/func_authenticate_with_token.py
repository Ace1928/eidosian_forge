import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
def authenticate_with_token(self, token, service_url=None):
    self.auth_token = token
    if not self.service_url:
        if not service_url:
            raise exceptions.ServiceUrlNotGiven()
        else:
            self.service_url = service_url