from __future__ import (absolute_import, division, print_function)
def check_for_0_or_1_records(api, response, error, query=None):
    """return None if no record was returned by the API
       return record if one record was returned by the API
       return error otherwise (error, no response, more than 1 record)
    """
    if error:
        return (None, api_error(api, error)) if api else (None, error)
    if not response:
        return (None, no_response_error(api, response))
    num_records = get_num_records(response)
    if num_records == 0:
        return (None, None)
    if num_records != 1:
        return unexpected_response_error(api, response, query)
    if 'records' in response:
        return (response['records'][0], None)
    return (response, None)