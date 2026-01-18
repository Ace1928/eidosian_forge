import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def delete_saml_provider(self, saml_provider_arn):
    """
        Deletes a SAML provider.

        Deleting the provider does not update any roles that reference
        the SAML provider as a principal in their trust policies. Any
        attempt to assume a role that references a SAML provider that
        has been deleted will fail.
        This operation requires `Signature Version 4`_.

        :type saml_provider_arn: string
        :param saml_provider_arn: The Amazon Resource Name (ARN) of the SAML
            provider to delete.

        """
    params = {'SAMLProviderArn': saml_provider_arn}
    return self.get_response('DeleteSAMLProvider', params)