from __future__ import (absolute_import, division, print_function)
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_0_or_more_records(rest_api, api, query=None, fields=None):
    query = build_query_with_fields(query, fields)
    response, error = rest_api.get(api, query)
    records, error = rrh.check_for_0_or_more_records(api, response, error)
    return (records, error)