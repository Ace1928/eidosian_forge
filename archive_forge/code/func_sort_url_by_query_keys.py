import copy
import io
import json
import testtools
from urllib import parse
from glanceclient.v2 import schemas
def sort_url_by_query_keys(url):
    """A helper function which sorts the keys of the query string of a url.

       For example, an input of '/v2/tasks?sort_key=id&sort_dir=asc&limit=10'
       returns '/v2/tasks?limit=10&sort_dir=asc&sort_key=id'. This is to
       prevent non-deterministic ordering of the query string causing
       problems with unit tests.
    :param url: url which will be ordered by query keys
    :returns url: url with ordered query keys
    """
    parsed = parse.urlparse(url)
    queries = parse.parse_qsl(parsed.query, True)
    sorted_query = sorted(queries, key=lambda x: x[0])
    encoded_sorted_query = parse.urlencode(sorted_query, True)
    url_parts = (parsed.scheme, parsed.netloc, parsed.path, parsed.params, encoded_sorted_query, parsed.fragment)
    return parse.urlunparse(url_parts)