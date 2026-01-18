import time
import threading
def process_scheduled_consumption(self, token):
    """Processes a scheduled consumption request that has completed

        :type token: RequestToken
        :param token: The token associated to the consumption
            request that is used to identify the request.
        """
    scheduled_retry = self._tokens_to_scheduled_consumption.pop(token)
    self._total_wait = max(self._total_wait - scheduled_retry['time_to_consume'], 0)