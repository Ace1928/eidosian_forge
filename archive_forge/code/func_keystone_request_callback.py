import copy
import json
from oslo_utils import uuidutils
def keystone_request_callback(request, context):
    context.headers['Content-Type'] = 'application/json'
    if request.url == BASE_URL:
        return V3_VERSION_LIST
    elif request.url == BASE_URL + '/v2.0':
        token_id, token_data = generate_v2_project_scoped_token()
        return token_data
    elif request.url.startswith('http://multiple.service.names'):
        token_id, token_data = generate_v2_project_scoped_token()
        return json.dumps(token_data)
    elif request.url == BASE_URL + '/v3':
        token_id, token_data = generate_v3_project_scoped_token()
        context.headers['X-Subject-Token'] = token_id
        context.status_code = 201
        return token_data
    elif 'wrongdiscoveryresponse.discovery.com' in request.url:
        return str(WRONG_VERSION_RESPONSE)
    else:
        context.status_code = 500
        return str(WRONG_VERSION_RESPONSE)