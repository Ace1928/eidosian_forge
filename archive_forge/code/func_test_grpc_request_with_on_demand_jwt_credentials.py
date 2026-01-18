import google.auth
import google.auth.credentials
import google.auth.jwt
import google.auth.transport.grpc
from google.oauth2 import service_account
from google.cloud import pubsub_v1
def test_grpc_request_with_on_demand_jwt_credentials():
    credentials, project_id = google.auth.default()
    credentials = google.auth.jwt.OnDemandCredentials.from_signing_credentials(credentials)
    client = pubsub_v1.PublisherClient(credentials=credentials)
    list_topics_iter = client.list_topics(project='projects/{}'.format(project_id))
    list(list_topics_iter)