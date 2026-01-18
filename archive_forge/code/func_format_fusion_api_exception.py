from __future__ import absolute_import, division, print_function
import sys
import json
import re
import traceback as trace
def format_fusion_api_exception(exception, traceback=None):
    """Formats `fusion.rest.ApiException` into a simple short form, suitable
    for Ansible error output. Returns a (message: str, body: dict) tuple."""
    message = None
    code = None
    resource_name = None
    request_id = None
    body = None
    call_site = _extract_rest_call_site(traceback)
    try:
        body = json.loads(exception.body)
        request_id = body.get('request_id', None)
        error = body['error']
        message = error.get('message')
        code = error.get('pure_code')
        if not code:
            code = exception.status
        if not code:
            code = error.get('http_code')
        resource_name = error['details']['name']
    except Exception:
        pass
    output = ''
    if call_site:
        output += "'{0}' failed".format(call_site)
    else:
        output += 'request failed'
    if message:
        output += ', {0}'.format(message.replace('"', "'"))
    details = DetailsPrinter(output)
    if resource_name:
        details.append("resource: '{0}'".format(resource_name))
    if code:
        details.append("code: '{0}'".format(code))
    if request_id:
        details.append("request id: '{0}'".format(request_id))
    output = details.finish()
    return (output, body)