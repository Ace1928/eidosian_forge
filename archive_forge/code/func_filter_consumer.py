import abc
import string
from keystone import exception
def filter_consumer(consumer_ref):
    """Filter out private items in a consumer dict.

    'secret' is never returned.

    :returns: consumer_ref

    """
    if consumer_ref:
        consumer_ref = consumer_ref.copy()
        consumer_ref.pop('secret', None)
    return consumer_ref