import base64
import hmac
import itertools
from collections import OrderedDict
from hashlib import md5
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.portal
import twisted.internet.protocol
import twisted.mail.pop3
import twisted.mail.protocols
from twisted import cred, internet, mail
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet import defer
from twisted.internet.testing import LineSendingProtocol
from twisted.mail import pop3
from twisted.protocols import loopback
from twisted.python import failure
from twisted.trial import unittest, util
def run_PASS(self, real_user, real_password, tried_user=None, tried_password=None, after_auth_input=[], after_auth_output=[]):
    """
        Test a login with PASS.

        If L{real_user} matches L{tried_user} and L{real_password} matches
        L{tried_password}, a successful login will be expected.
        Otherwise an unsuccessful login will be expected.

        @type real_user: L{bytes}
        @param real_user: The user to test.

        @type real_password: L{bytes}
        @param real_password: The password of the test user.

        @type tried_user: L{bytes} or L{None}
        @param tried_user: The user to call USER with.
            If None, real_user will be used.

        @type tried_password: L{bytes} or L{None}
        @param tried_password: The password to call PASS with.
            If None, real_password will be used.

        @type after_auth_input: L{list} of l{bytes}
        @param after_auth_input: Extra protocol input after authentication.

        @type after_auth_output: L{list} of l{bytes}
        @param after_auth_output: Extra protocol output after authentication.
        """
    if not tried_user:
        tried_user = real_user
    if not tried_password:
        tried_password = real_password
    response = [b'+OK <moshez>', b'+OK USER accepted, send PASS', b'-ERR Authentication failed']
    if real_user == tried_user and real_password == tried_password:
        response = [b'+OK <moshez>', b'+OK USER accepted, send PASS', b'+OK Authentication succeeded']
    fullInput = [b' '.join([b'USER', tried_user]), b' '.join([b'PASS', tried_password])]
    fullInput += after_auth_input + [b'QUIT']
    response += after_auth_output + [b'+OK ']
    return self.runTest(fullInput, response, protocolInstance=DummyPOP3Auth(real_user, real_password))