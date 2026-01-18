from oslo_utils import uuidutils
from urllib.parse import parse_qs
from urllib.parse import urlparse
from designateclient import exceptions
def parse_query_from_url(url):
    """
    Helper to get key bits of data from the "next" url returned
    from the API on collections
    :param url:
    :return: dict
    """
    values = parse_qs(urlparse(url)[4])
    return {k: values[k][0] for k in values.keys()}