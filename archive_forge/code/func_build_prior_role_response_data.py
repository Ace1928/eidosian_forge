from keystone.common import provider_api
from keystone.server import flask as ks_flask
def build_prior_role_response_data(prior_role_id, prior_role_name):
    return {'id': prior_role_id, 'links': {'self': ks_flask.base_url(path='/roles/%s' % prior_role_id)}, 'name': prior_role_name}