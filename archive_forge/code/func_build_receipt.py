import flask
import http.client
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import provider_api
from keystone import exception
def build_receipt(mfa_error):
    receipt = PROVIDERS.receipt_provider_api.issue_receipt(mfa_error.user_id, mfa_error.methods)
    resp_data = _render_receipt_response_from_model(receipt)
    resp_body = jsonutils.dumps(resp_data)
    response = flask.make_response(resp_body, http.client.UNAUTHORIZED)
    response.headers[authorization.AUTH_RECEIPT_HEADER] = receipt.id
    response.headers['Content-Type'] = 'application/json'
    return response