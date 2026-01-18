import time
import threading
def is_scheduled(self, token):
    """Indicates if a consumption request has been scheduled

        :type token: RequestToken
        :param token: The token associated to the consumption
            request that is used to identify the request.
        """
    return token in self._tokens_to_scheduled_consumption