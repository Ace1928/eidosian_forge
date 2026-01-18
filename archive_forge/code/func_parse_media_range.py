from __future__ import absolute_import
from functools import reduce
import six
def parse_media_range(range):
    """Parse a media-range into its component parts.

    Carves up a media range and returns a tuple of the (type, subtype,
    params) where 'params' is a dictionary of all the parameters for the media
    range.  For example, the media range 'application/*;q=0.5' would get parsed
    into:

       ('application', '*', {'q', '0.5'})

    In addition this function also guarantees that there is a value for 'q'
    in the params dictionary, filling it in with a proper default if
    necessary.
    """
    type, subtype, params = parse_mime_type(range)
    if 'q' not in params or not params['q'] or (not float(params['q'])) or (float(params['q']) > 1) or (float(params['q']) < 0):
        params['q'] = '1'
    return (type, subtype, params)