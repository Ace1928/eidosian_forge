import abc
import string
from keystone import exception
@abc.abstractmethod
def authorize_request_token(self, request_token_id, user_id, role_ids):
    """Authorize request token.

        :param request_token_id: the id of the request token, to be authorized
        :type request_token_id: string
        :param user_id: the id of the authorizing user
        :type user_id: string
        :param role_ids: list of role ids to authorize
        :type role_ids: list
        :returns: verifier

        """
    raise exception.NotImplemented()