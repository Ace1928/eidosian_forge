import google
from google.auth import impersonated_credentials
import google.auth.transport.requests
def idtoken_from_impersonated_credentials(impersonated_service_account: str, scope: str, target_audience: str):
    """
      Use a service account (SA1) to impersonate as another service account (SA2) and obtain id token
      for the impersonated account.
      To obtain token for SA2, SA1 should have the "roles/iam.serviceAccountTokenCreator" permission
      on SA2.

    Args:
        impersonated_service_account: The name of the privilege-bearing service account for whom the credential is created.
            Examples: name@project.service.gserviceaccount.com

        scope: Provide the scopes that you might need to request to access Google APIs,
            depending on the level of access you need.
            For this example, we use the cloud-wide scope and use IAM to narrow the permissions.
            https://cloud.google.com/docs/authentication#authorization_for_services
            For more information, see: https://developers.google.com/identity/protocols/oauth2/scopes

        target_audience: The service name for which the id token is requested. Service name refers to the
            logical identifier of an API service, such as "iap.googleapis.com".
            Examples: iap.googleapis.com
    """
    credentials, project_id = google.auth.default()
    target_credentials = impersonated_credentials.Credentials(source_credentials=credentials, target_principal=impersonated_service_account, delegates=[], target_scopes=[scope], lifetime=300)
    id_creds = impersonated_credentials.IDTokenCredentials(target_credentials, target_audience=target_audience, include_email=True)
    request = google.auth.transport.requests.Request()
    id_creds.refresh(request)
    print('Generated ID token.')