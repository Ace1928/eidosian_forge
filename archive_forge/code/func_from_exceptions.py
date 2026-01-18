import multiprocessing
import requests
from . import thread
from .._compat import queue
@classmethod
def from_exceptions(cls, exceptions, **kwargs):
    """Create a :class:`~Pool` from an :class:`~ThreadException`\\ s.

        Provided an iterable that provides :class:`~ThreadException` objects,
        this classmethod will generate a new pool to retry the requests that
        caused the exceptions.

        :param exceptions:
            Iterable that returns :class:`~ThreadException`
        :type exceptions: iterable
        :param kwargs:
            Keyword arguments passed to the :class:`~Pool` initializer.
        :returns: An initialized :class:`~Pool` object.
        :rtype: :class:`~Pool`
        """
    job_queue = queue.Queue()
    for exc in exceptions:
        job_queue.put(exc.request_kwargs)
    return cls(job_queue=job_queue, **kwargs)