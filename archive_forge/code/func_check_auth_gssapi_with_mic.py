import threading
from paramiko import util
from paramiko.common import (
def check_auth_gssapi_with_mic(self, username, gss_authenticated=AUTH_FAILED, cc_file=None):
    """
        Authenticate the given user to the server if he is a valid krb5
        principal.

        :param str username: The username of the authenticating client
        :param int gss_authenticated: The result of the krb5 authentication
        :param str cc_filename: The krb5 client credentials cache filename
        :return: ``AUTH_FAILED`` if the user is not authenticated otherwise
                 ``AUTH_SUCCESSFUL``
        :rtype: int
        :note: Kerberos credential delegation is not supported.
        :see: `.ssh_gss`
        :note: : We are just checking in L{AuthHandler} that the given user is
                 a valid krb5 principal!
                 We don't check if the krb5 principal is allowed to log in on
                 the server, because there is no way to do that in python. So
                 if you develop your own SSH server with paramiko for a certain
                 platform like Linux, you should call C{krb5_kuserok()} in
                 your local kerberos library to make sure that the
                 krb5_principal has an account on the server and is allowed to
                 log in as a user.
        :see: http://www.unix.com/man-page/all/3/krb5_kuserok/
        """
    if gss_authenticated == AUTH_SUCCESSFUL:
        return AUTH_SUCCESSFUL
    return AUTH_FAILED