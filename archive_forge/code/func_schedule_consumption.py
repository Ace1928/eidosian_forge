import time
import threading
def schedule_consumption(self, amt, token, time_to_consume):
    """Schedules a wait time to be able to consume an amount

        :type amt: int
        :param amt: The amount of bytes scheduled to be consumed

        :type token: RequestToken
        :param token: The token associated to the consumption
            request that is used to identify the request.

        :type time_to_consume: float
        :param time_to_consume: The desired time it should take for that
            specific request amount to be consumed in regardless of previously
            scheduled consumption requests

        :rtype: float
        :returns: The amount of time to wait for the specific request before
            actually consuming the specified amount.
        """
    self._total_wait += time_to_consume
    self._tokens_to_scheduled_consumption[token] = {'wait_duration': self._total_wait, 'time_to_consume': time_to_consume}
    return self._total_wait