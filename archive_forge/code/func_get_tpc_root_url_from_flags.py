import json
import logging
from typing import Dict, NamedTuple, Optional, Union
import urllib
from absl import flags
from utils import bq_consts
from utils import bq_error
def get_tpc_root_url_from_flags(service: Service, inputted_flags: NamedTuple('InputtedFlags', [('API', flags.FlagHolder[Optional[str]]), ('UNIVERSE_DOMAIN', flags.FlagHolder[Optional[str]]), ('LOCATION', flags.FlagHolder[Optional[str]]), ('USE_LEP', flags.FlagHolder[bool]), ('USE_REP', flags.FlagHolder[bool]), ('USE_REGIONAL_ENDPOINTS', flags.FlagHolder[bool]), ('MTLS', flags.FlagHolder[bool])]), local_params: Optional[NamedTuple('LocalParams', [])]=None) -> str:
    """Takes BQ CLI flags to build a root URL to make requests to.

  If the `api` flag is set, and is a http/https URL then it will be used
  otherwise the result is built up from the different options for a TPC service
  endpoint.

  Args:
    service: The service that this request will be made to. Usually the API
      that is being hit.
    inputted_flags: The flags set, usually straight from bq_flags.

  Returns:
    The root URL to be used for BQ requests. This is built from the service
    being targeted and a number of flags as arguments. It's intended to be used
    both for building the URL to request the discovery doc from, and to override
    the rootUrl and servicePath values of the discovery doc when they're
    incorrect. It always ends with a trailing slash.

  Raises:
    BigqueryClientError: If the flags are used incorrectly.
  """
    number_of_flags_requesting_a_regional_api = [inputted_flags.USE_LEP.value, inputted_flags.USE_REP.value, inputted_flags.USE_REGIONAL_ENDPOINTS.value].count(True)
    if number_of_flags_requesting_a_regional_api > 1:
        raise bq_error.BigqueryClientError('Only one of use_lep, use_rep or use_regional_endpoints can be used at a time')
    if number_of_flags_requesting_a_regional_api == 1 and (not inputted_flags.LOCATION.value):
        raise bq_error.BigqueryClientError('A region is needed when the use_lep, use_rep or use_regional_endpoints flags are used.')
    if inputted_flags.API.present:
        logging.info('Looking for a root URL and an `api` value was found, using that: %s', inputted_flags.API.value)
        return add_trailing_slash_if_missing(inputted_flags.API.value)
    if number_of_flags_requesting_a_regional_api == 0 and inputted_flags.LOCATION.value:
        region = None
    else:
        region = inputted_flags.LOCATION.value
    if inputted_flags.USE_REGIONAL_ENDPOINTS.value:
        logging.info('Building a root URL and `use_regional_endpoints` is present, forcing LEP')
        is_lep = True
    else:
        is_lep = inputted_flags.USE_LEP.value
    if inputted_flags.UNIVERSE_DOMAIN.value:
        universe_domain = inputted_flags.UNIVERSE_DOMAIN.value
    else:
        universe_domain = 'googleapis.com'
    hostname = _get_tpc_service_endpoint_hostname(service=service, universe_domain=universe_domain, region=region, is_mtls=inputted_flags.MTLS.value, is_rep=inputted_flags.USE_REP.value, is_lep=is_lep)
    root_url = add_trailing_slash_if_missing(urllib.parse.urlunsplit(urllib.parse.SplitResult(scheme='https', netloc=hostname, path='', query='', fragment='')))
    logging.info('Final root URL built as: %s', root_url)
    return root_url