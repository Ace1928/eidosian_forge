import threading
from paramiko import util
from paramiko.common import (
def check_auth_interactive(self, username, submethods):
    """
        Begin an interactive authentication challenge, if supported.  You
        should override this method in server mode if you want to support the
        ``"keyboard-interactive"`` auth type, which requires you to send a
        series of questions for the client to answer.

        Return ``AUTH_FAILED`` if this auth method isn't supported.  Otherwise,
        you should return an `.InteractiveQuery` object containing the prompts
        and instructions for the user.  The response will be sent via a call
        to `check_auth_interactive_response`.

        The default implementation always returns ``AUTH_FAILED``.

        :param str username: the username of the authenticating client
        :param str submethods:
            a comma-separated list of methods preferred by the client (usually
            empty)
        :return:
            ``AUTH_FAILED`` if this auth method isn't supported; otherwise an
            object containing queries for the user
        :rtype: int or `.InteractiveQuery`
        """
    return AUTH_FAILED