import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
def opt_help_auth_type(self, authType):
    """
        Show help for a particular authentication type.
        """
    try:
        cf = findCheckerFactory(authType)
    except InvalidAuthType:
        raise usage.UsageError('Invalid auth type: %s' % authType)
    self.authOutput.write('Usage: --auth %s[:ArgString]\n' % authType)
    self.authOutput.write('ArgString format: %s\n' % cf.argStringFormat)
    self.authOutput.write('\n')
    for line in cf.authHelp.strip().splitlines():
        self.authOutput.write('  %s\n' % line.rstrip())
    self.authOutput.write('\n')
    if not self.supportsCheckerFactory(cf):
        self.authOutput.write('  %s\n' % notSupportedWarning)
        self.authOutput.write('\n')
    raise SystemExit(0)