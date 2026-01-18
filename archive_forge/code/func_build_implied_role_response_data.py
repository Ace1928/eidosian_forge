from keystone.common import provider_api
from keystone.server import flask as ks_flask
def build_implied_role_response_data(implied_role):
    return {'id': implied_role['id'], 'links': {'self': ks_flask.base_url(path='/roles/%s' % implied_role['id'])}, 'name': implied_role['name']}