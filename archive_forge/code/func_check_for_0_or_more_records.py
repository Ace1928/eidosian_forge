from __future__ import (absolute_import, division, print_function)
def check_for_0_or_more_records(api, response, error):
    """return None if no record was returned by the API
       return records if one or more records was returned by the API
       return error otherwise (error, no response)
    """
    if error:
        return (None, api_error(api, error)) if api else (None, error)
    if not response:
        return (None, no_response_error(api, response))
    if get_num_records(response) == 0:
        return (None, None)
    if 'records' in response:
        return (response['records'], None)
    error = 'No "records" key in %s' % response
    return (None, api_error(api, error)) if api else (None, error)