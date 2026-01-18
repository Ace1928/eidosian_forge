import json
from six.moves import urllib
from google_reauth import errors
def send_challenge_result(http_request, session_id, challenge_id, client_input, access_token):
    """Attempt to refresh access token by sending next challenge result.

    Args:
        http_request (Callable): callable to run http requests. Accepts uri,
            method, body and headers. Returns a tuple: (response, content)
        session_id (str): session id returned by the initial reauth call.
        challenge_id (str): challenge id returned by the initial reauth call.
        client_input: dict with a challenge-specific client input. For example:
            ``{'credential': password}`` for password challenge.
        access_token (str): Access token with reauth scopes.

    Returns:
        dict: The response from the reauth API.
    """
    body = {'sessionId': session_id, 'challengeId': challenge_id, 'action': 'RESPOND', 'proposalResponse': client_input}
    return _endpoint_request(http_request, '/{0}:continue'.format(session_id), body, access_token)