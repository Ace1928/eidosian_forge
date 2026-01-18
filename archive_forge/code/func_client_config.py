from pprint import pformat
from six import iteritems
import re
@client_config.setter
def client_config(self, client_config):
    """
        Sets the client_config of this V1beta1Webhook.
        ClientConfig defines how to communicate with the hook. Required

        :param client_config: The client_config of this V1beta1Webhook.
        :type: AdmissionregistrationV1beta1WebhookClientConfig
        """
    if client_config is None:
        raise ValueError('Invalid value for `client_config`, must not be `None`')
    self._client_config = client_config