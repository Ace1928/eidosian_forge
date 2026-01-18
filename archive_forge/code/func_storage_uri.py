from boto.pyami.config import Config, BotoConfigLocations
from boto.storage_uri import BucketStorageUri, FileStorageUri
import boto.plugin
import datetime
import os
import platform
import re
import sys
import logging
import logging.config
from boto.compat import urlparse
from boto.exception import InvalidUriError
def storage_uri(uri_str, default_scheme='file', debug=0, validate=True, bucket_storage_uri_class=BucketStorageUri, suppress_consec_slashes=True, is_latest=False):
    """
    Instantiate a StorageUri from a URI string.

    :type uri_str: string
    :param uri_str: URI naming bucket + optional object.
    :type default_scheme: string
    :param default_scheme: default scheme for scheme-less URIs.
    :type debug: int
    :param debug: debug level to pass in to boto connection (range 0..2).
    :type validate: bool
    :param validate: whether to check for bucket name validity.
    :type bucket_storage_uri_class: BucketStorageUri interface.
    :param bucket_storage_uri_class: Allows mocking for unit tests.
    :param suppress_consec_slashes: If provided, controls whether
        consecutive slashes will be suppressed in key paths.
    :type is_latest: bool
    :param is_latest: whether this versioned object represents the
        current version.

    We allow validate to be disabled to allow caller
    to implement bucket-level wildcarding (outside the boto library;
    see gsutil).

    :rtype: :class:`boto.StorageUri` subclass
    :return: StorageUri subclass for given URI.

    ``uri_str`` must be one of the following formats:

    * gs://bucket/name
    * gs://bucket/name#ver
    * s3://bucket/name
    * gs://bucket
    * s3://bucket
    * filename (which could be a Unix path like /a/b/c or a Windows path like
      C:\x07\x08\\c)

    The last example uses the default scheme ('file', unless overridden).
    """
    version_id = None
    generation = None
    end_scheme_idx = uri_str.find('://')
    if end_scheme_idx == -1:
        scheme = default_scheme.lower()
        path = uri_str
    else:
        scheme = uri_str[0:end_scheme_idx].lower()
        path = uri_str[end_scheme_idx + 3:]
    if scheme not in ['file', 's3', 'gs']:
        raise InvalidUriError('Unrecognized scheme "%s"' % scheme)
    if scheme == 'file':
        is_stream = False
        if path == '-':
            is_stream = True
        return FileStorageUri(path, debug, is_stream)
    else:
        path_parts = path.split('/', 1)
        bucket_name = path_parts[0]
        object_name = ''
        if validate and bucket_name and (not BUCKET_NAME_RE.match(bucket_name) or TOO_LONG_DNS_NAME_COMP.search(bucket_name)):
            raise InvalidUriError('Invalid bucket name in URI "%s"' % uri_str)
        if scheme == 'gs':
            match = GENERATION_RE.search(path)
            if match:
                md = match.groupdict()
                versionless_uri_str = md['versionless_uri_str']
                path_parts = versionless_uri_str.split('/', 1)
                generation = int(md['generation'])
        elif scheme == 's3':
            match = VERSION_RE.search(path)
            if match:
                md = match.groupdict()
                versionless_uri_str = md['versionless_uri_str']
                path_parts = versionless_uri_str.split('/', 1)
                version_id = md['version_id']
        else:
            raise InvalidUriError('Unrecognized scheme "%s"' % scheme)
        if len(path_parts) > 1:
            object_name = path_parts[1]
        return bucket_storage_uri_class(scheme, bucket_name, object_name, debug, suppress_consec_slashes=suppress_consec_slashes, version_id=version_id, generation=generation, is_latest=is_latest)