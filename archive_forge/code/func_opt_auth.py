import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
def opt_auth(self, description):
    """
        Specify an authentication method for the server.
        """
    try:
        self.addChecker(makeChecker(description))
    except UnsupportedInterfaces as e:
        raise usage.UsageError('Auth plugin not supported: %s' % e.args[0])
    except InvalidAuthType as e:
        raise usage.UsageError('Auth plugin not recognized: %s' % e.args[0])
    except Exception as e:
        raise usage.UsageError('Unexpected error: %s' % e)