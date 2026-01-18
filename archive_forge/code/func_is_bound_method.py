import inspect
import urllib.parse as urlparse  # noqa
from urllib.parse import quote, unquote_plus  # noqa
from urllib.request import urlopen, URLError  # noqa
from html import escape  # noqa
def is_bound_method(ob):
    return inspect.ismethod(ob) and ob.__self__ is not None