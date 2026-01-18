import json
from six.moves import http_client
from six.moves import urllib
from google.oauth2 import utils
def exchange_token(self, request, grant_type, subject_token, subject_token_type, resource=None, audience=None, scopes=None, requested_token_type=None, actor_token=None, actor_token_type=None, additional_options=None, additional_headers=None):
    """Exchanges the provided token for another type of token based on the
        rfc8693 spec.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            grant_type (str): The OAuth 2.0 token exchange grant type.
            subject_token (str): The OAuth 2.0 token exchange subject token.
            subject_token_type (str): The OAuth 2.0 token exchange subject token type.
            resource (Optional[str]): The optional OAuth 2.0 token exchange resource field.
            audience (Optional[str]): The optional OAuth 2.0 token exchange audience field.
            scopes (Optional[Sequence[str]]): The optional list of scopes to use.
            requested_token_type (Optional[str]): The optional OAuth 2.0 token exchange requested
                token type.
            actor_token (Optional[str]): The optional OAuth 2.0 token exchange actor token.
            actor_token_type (Optional[str]): The optional OAuth 2.0 token exchange actor token type.
            additional_options (Optional[Mapping[str, str]]): The optional additional
                non-standard Google specific options.
            additional_headers (Optional[Mapping[str, str]]): The optional additional
                headers to pass to the token exchange endpoint.

        Returns:
            Mapping[str, str]: The token exchange JSON-decoded response data containing
                the requested token and its expiration time.

        Raises:
            google.auth.exceptions.OAuthError: If the token endpoint returned
                an error.
        """
    request_body = {'grant_type': grant_type, 'resource': resource, 'audience': audience, 'scope': ' '.join(scopes or []), 'requested_token_type': requested_token_type, 'subject_token': subject_token, 'subject_token_type': subject_token_type, 'actor_token': actor_token, 'actor_token_type': actor_token_type, 'options': None}
    if additional_options:
        request_body['options'] = urllib.parse.quote(json.dumps(additional_options))
    for k, v in dict(request_body).items():
        if v is None or v == '':
            del request_body[k]
    return self._make_request(request, additional_headers, request_body)