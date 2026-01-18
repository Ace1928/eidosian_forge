from __future__ import absolute_import, division, print_function
def get_poller_result(self, poller, timeout):
    try:
        poller.wait(timeout=timeout)
        return poller.result()
    except Exception as exc:
        raise