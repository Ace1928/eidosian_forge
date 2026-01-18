import abc
import string
from keystone import exception
@abc.abstractmethod
def get_request_token(self, request_token_id):
    """Get request token.

        :param request_token_id: the id of the request token
        :type request_token_id: string
        :returns: request_token_ref

        """
    raise exception.NotImplemented()