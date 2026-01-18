import logging
from logging.handlers import BufferingHandler
def matchall(self, kwarglist):
    """
        Accept a list of keyword argument values and ensure that the handler's
        buffer of stored records matches the list one-for-one.

        Return `True` if exactly matched, else `False`.

        :param kwarglist: A list of keyword-argument dictionaries, each of
                          which will be passed to :meth:`matches` with the
                          corresponding record from the buffer.
        """
    if self.count != len(kwarglist):
        result = False
    else:
        result = True
        for d, kwargs in zip(self.buffer, kwarglist):
            if not self.matcher.matches(d, **kwargs):
                result = False
                break
    return result