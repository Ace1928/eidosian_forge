import http.client as http
import time
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import requests
def poll_entity(url, headers, callback, max_sec=10, delay_sec=0.2, require_success=True):
    """Poll a given URL passing the parsed entity to a callback.

    This is a utility method that repeatedly GETs a URL, and calls
    a callback with the result. The callback determines if we should
    keep polling by returning True (up to the timeout).

    :param url: The url to fetch
    :param headers: The request headers to use for the fetch
    :param callback: A function that takes the parsed entity and is expected
                     to return True if we should keep polling
    :param max_sec: The overall timeout before we fail
    :param delay_sec: The time between fetches
    :param require_success: Assert resp_code is http.OK each time before
                            calling the callback
    """
    timer = timeutils.StopWatch(max_sec)
    timer.start()
    while not timer.expired():
        resp = requests.get(url, headers=headers)
        if require_success and resp.status_code != http.OK:
            raise Exception('Received %i response from server' % resp.status_code)
        entity = resp.json()
        keep_polling = callback(entity)
        if keep_polling is not True:
            return keep_polling
        time.sleep(delay_sec)
    raise Exception('Poll timeout if %i seconds exceeded!' % max_sec)