import logging
import ssl
import sys
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, Struct
def get_http_wrapper(library=None, features=[]):
    if library is not None:
        try:
            return _http_connectors[library]
        except KeyError:
            raise RuntimeError('%s transport is not available' % (library,))
    if not features:
        return _http_connectors.get('httplib2', _http_connectors['urllib2'])
    current_candidates = _http_connectors.keys()
    new_candidates = []
    for feature in features:
        for candidate in current_candidates:
            if candidate in _http_facilities.get(feature, []):
                new_candidates.append(candidate)
        current_candidates = new_candidates
        new_candidates = []
    try:
        candidate_name = current_candidates[0]
    except IndexError:
        raise RuntimeError('no transport available which supports these features: %s' % (features,))
    else:
        return _http_connectors[candidate_name]