import sys
import fixtures
from functools import wraps
@response_timeout.setter
def response_timeout(self, value):
    self.conf.set_override('rpc_response_timeout', value)