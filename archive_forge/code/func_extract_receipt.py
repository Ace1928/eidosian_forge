import flask
import http.client
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import provider_api
from keystone import exception
def extract_receipt(auth_context):
    receipt_id = flask.request.headers.get(authorization.AUTH_RECEIPT_HEADER, None)
    if receipt_id:
        receipt = PROVIDERS.receipt_provider_api.validate_receipt(receipt_id)
        if auth_context['user_id'] != receipt.user_id:
            raise exception.ReceiptNotFound('AuthContext user_id: %s does not match user_id for supplied auth receipt: %s' % (auth_context['user_id'], receipt.user_id), receipt_id=receipt_id)
    else:
        receipt = None
    return receipt