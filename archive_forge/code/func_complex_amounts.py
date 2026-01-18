import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
def complex_amounts(*fields):

    def decorator(func):

        def wrapper(self, *args, **kw):
            for field in filter(kw.has_key, fields):
                amount = kw.pop(field)
                kw[field + '.Value'] = getattr(amount, 'Value', str(amount))
                kw[field + '.CurrencyCode'] = getattr(amount, 'CurrencyCode', self.currencycode)
            return func(self, *args, **kw)
        wrapper.__doc__ = '{0}\nComplex Amounts: {1}'.format(func.__doc__, ', '.join(fields))
        return add_attrs_from(func, to=wrapper)
    return decorator