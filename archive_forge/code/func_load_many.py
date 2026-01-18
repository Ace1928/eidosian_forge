from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
def load_many(self, keys):
    """
        Loads multiple keys, promising an array of values

        >>> a, b = await my_loader.load_many([ 'a', 'b' ])

        This is equivalent to the more verbose:

        >>> a, b = await Promise.all([
        >>>    my_loader.load('a'),
        >>>    my_loader.load('b')
        >>> ])
        """
    if not isinstance(keys, Iterable):
        raise TypeError(('The loader.loadMany() function must be called with Array<key> ' + 'but got: {}.').format(keys))
    return Promise.all([self.load(key) for key in keys])