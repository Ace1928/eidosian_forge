from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.strcred import ICheckerFactory
from twisted.internet import defer
def checkPwd(self, pwd, username, password):
    """
        Obtain the encrypted password for C{username} from the Unix password
        database using L{pwd.getpwnam}, and see if it it matches it matches
        C{password}.

        @param pwd: Module which provides functions which
                    access to the Unix password database.
        @type pwd: C{module}
        @param username: The user to look up in the Unix password database.
        @type username: L{unicode}/L{str} or L{bytes}
        @param password: The password to compare.
        @type username: L{unicode}/L{str} or L{bytes}
        """
    try:
        if isinstance(username, bytes):
            username = username.decode('utf-8')
        cryptedPass = pwd.getpwnam(username).pw_passwd
    except KeyError:
        return defer.fail(UnauthorizedLogin())
    else:
        if cryptedPass in ('*', 'x'):
            return None
        elif verifyCryptedPassword(cryptedPass, password):
            return defer.succeed(username)