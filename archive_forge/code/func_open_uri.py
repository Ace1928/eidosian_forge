import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def open_uri(uri, mode, transport_params):
    deprecated = ('multipart_upload_kwargs', 'object_kwargs', 'resource', 'resource_kwargs', 'session', 'singlepart_upload_kwargs')
    detected = [k for k in deprecated if k in transport_params]
    if detected:
        doc_url = 'https://github.com/RaRe-Technologies/smart_open/blob/develop/MIGRATING_FROM_OLDER_VERSIONS.rst'
        message = 'ignoring the following deprecated transport parameters: %r. See <%s> for details' % (detected, doc_url)
        warnings.warn(message, UserWarning)
    parsed_uri = parse_uri(uri)
    parsed_uri, transport_params = _consolidate_params(parsed_uri, transport_params)
    kwargs = smart_open.utils.check_kwargs(open, transport_params)
    return open(parsed_uri['bucket_id'], parsed_uri['key_id'], mode, **kwargs)