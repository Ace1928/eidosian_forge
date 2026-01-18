from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import urllib.parse
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import requests
def get_signed_url(client_id, duration, headers, host, key, verb, parameters, path, region, delegates):
    """Gets a signed URL for a GCS XML API request.

  https://cloud.google.com/storage/docs/access-control/signed-urls

  Args:
    client_id (str): Email of the service account that makes the request.
    duration (int): Amount of time (seconds) that the URL is valid for.
    headers (dict[str, str]): User-inputted headers for the request.
    host (str): The endpoint URL for the request. This should include a scheme,
      e.g. "https://"
    key (crypto.PKey): Key for the service account specified by client_id.
    verb (str): HTTP verb associated with the request.
    parameters (dict[str, str]): User-inputted parameters for the request.
    path (str): Of the form `/bucket-name/object-name`. Specifies the resource
      that is targeted by the request.
    region (str): The region of the target resource instance.
    delegates (list[str]|None): The list of service accounts in a delegation
      chain specified in --impersonate-service-account.

  Returns:
    A URL (str) used to make the specified request.
  """
    encoded_path = urllib.parse.quote(path, safe='/~')
    signing_time = times.Now(tzinfo=times.UTC)
    _, _, host_without_scheme = host.rpartition('://')
    headers_to_sign = {'host': host_without_scheme}
    headers_to_sign.update(headers)
    canonical_headers_string = ''.join(['{}:{}\n'.format(k.lower(), v) for k, v in sorted(headers_to_sign.items())])
    canonical_signed_headers_string = ';'.join(sorted(headers_to_sign.keys()))
    canonical_scope = '{date}/{region}/storage/goog4_request'.format(date=signing_time.strftime('%Y%m%d'), region=region.lower())
    canonical_time = signing_time.strftime('%Y%m%dT%H%M%SZ')
    query_params_to_sign = {'x-goog-algorithm': _SIGNING_ALGORITHM, 'x-goog-credential': client_id + '/' + canonical_scope, 'x-goog-date': canonical_time, 'x-goog-signedheaders': canonical_signed_headers_string, 'x-goog-expires': str(duration)}
    query_params_to_sign.update(parameters)
    canonical_query_string = '&'.join(['{}={}'.format(k, urllib.parse.quote_plus(v)) for k, v in sorted(query_params_to_sign.items())])
    canonical_request_string = '\n'.join([verb, encoded_path, canonical_query_string, canonical_headers_string, canonical_signed_headers_string, _UNSIGNED_PAYLOAD])
    log.debug('Canonical request string:\n' + canonical_request_string)
    canonical_request_hash = hashlib.sha256(canonical_request_string.encode('utf-8')).hexdigest()
    string_to_sign = '\n'.join([_SIGNING_ALGORITHM, canonical_time, canonical_scope, canonical_request_hash])
    log.debug('String to sign:\n' + string_to_sign)
    raw_signature = _sign_with_key(key, string_to_sign) if key else _sign_with_iam(client_id, string_to_sign, delegates)
    signature = base64.b16encode(raw_signature).lower().decode('utf-8')
    return '{host}{path}?x-goog-signature={signature}&{query_string}'.format(host=host, path=encoded_path, signature=signature, query_string=canonical_query_string)