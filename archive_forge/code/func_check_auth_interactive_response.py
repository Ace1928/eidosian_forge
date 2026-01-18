import threading
from paramiko import util
from paramiko.common import (
def check_auth_interactive_response(self, responses):
    """
        Continue or finish an interactive authentication challenge, if
        supported.  You should override this method in server mode if you want
        to support the ``"keyboard-interactive"`` auth type.

        Return ``AUTH_FAILED`` if the responses are not accepted,
        ``AUTH_SUCCESSFUL`` if the responses are accepted and complete
        the authentication, or ``AUTH_PARTIALLY_SUCCESSFUL`` if your
        authentication is stateful, and this set of responses is accepted for
        authentication, but more authentication is required.  (In this latter
        case, `get_allowed_auths` will be called to report to the client what
        options it has for continuing the authentication.)

        If you wish to continue interactive authentication with more questions,
        you may return an `.InteractiveQuery` object, which should cause the
        client to respond with more answers, calling this method again.  This
        cycle can continue indefinitely.

        The default implementation always returns ``AUTH_FAILED``.

        :param responses: list of `str` responses from the client
        :return:
            ``AUTH_FAILED`` if the authentication fails; ``AUTH_SUCCESSFUL`` if
            it succeeds; ``AUTH_PARTIALLY_SUCCESSFUL`` if the interactive auth
            is successful, but authentication must continue; otherwise an
            object containing queries for the user
        :rtype: int or `.InteractiveQuery`
        """
    return AUTH_FAILED