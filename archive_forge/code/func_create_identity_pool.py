import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def create_identity_pool(self, identity_pool_name, allow_unauthenticated_identities, supported_login_providers=None, developer_provider_name=None, open_id_connect_provider_ar_ns=None):
    """
        Creates a new identity pool. The identity pool is a store of
        user identity information that is specific to your AWS
        account. The limit on identity pools is 60 per account.

        :type identity_pool_name: string
        :param identity_pool_name: A string that you provide.

        :type allow_unauthenticated_identities: boolean
        :param allow_unauthenticated_identities: TRUE if the identity pool
            supports unauthenticated logins.

        :type supported_login_providers: map
        :param supported_login_providers: Optional key:value pairs mapping
            provider names to provider app IDs.

        :type developer_provider_name: string
        :param developer_provider_name: The "domain" by which Cognito will
            refer to your users. This name acts as a placeholder that allows
            your backend and the Cognito service to communicate about the
            developer provider. For the `DeveloperProviderName`, you can use
            letters as well as period ( `.`), underscore ( `_`), and dash (
            `-`).
        Once you have set a developer provider name, you cannot change it.
            Please take care in setting this parameter.

        :type open_id_connect_provider_ar_ns: list
        :param open_id_connect_provider_ar_ns:

        """
    params = {'IdentityPoolName': identity_pool_name, 'AllowUnauthenticatedIdentities': allow_unauthenticated_identities}
    if supported_login_providers is not None:
        params['SupportedLoginProviders'] = supported_login_providers
    if developer_provider_name is not None:
        params['DeveloperProviderName'] = developer_provider_name
    if open_id_connect_provider_ar_ns is not None:
        params['OpenIdConnectProviderARNs'] = open_id_connect_provider_ar_ns
    return self.make_request(action='CreateIdentityPool', body=json.dumps(params))