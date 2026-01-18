import multiprocessing
import requests
from . import thread
from .._compat import queue
@classmethod
def from_urls(cls, urls, request_kwargs=None, **kwargs):
    """Create a :class:`~Pool` from an iterable of URLs.

        :param urls:
            Iterable that returns URLs with which we create a pool.
        :type urls: iterable
        :param dict request_kwargs:
            Dictionary of other keyword arguments to provide to the request
            method.
        :param kwargs:
            Keyword arguments passed to the :class:`~Pool` initializer.
        :returns: An initialized :class:`~Pool` object.
        :rtype: :class:`~Pool`
        """
    request_dict = {'method': 'GET'}
    request_dict.update(request_kwargs or {})
    job_queue = queue.Queue()
    for url in urls:
        job = request_dict.copy()
        job.update({'url': url})
        job_queue.put(job)
    return cls(job_queue=job_queue, **kwargs)