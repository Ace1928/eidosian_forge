from keystone.common import provider_api
from keystone.server import flask as ks_flask
def role_inference_response(prior_role_id):
    prior_role = PROVIDERS.role_api.get_role(prior_role_id)
    response = {'role_inference': {'prior_role': build_prior_role_response_data(prior_role_id, prior_role['name'])}}
    return response