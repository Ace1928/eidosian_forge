import requests
from wandb_gql import gql
import wandb
from wandb.apis.attrs import Attrs
def generate_api_key(self, description=None):
    """Generate a new api key.

        Returns:
            The new api key, or None on failure
        """
    try:
        key = self.user_api.client.execute(self.GENERATE_API_KEY_MUTATION, {'description': description})['generateApiKey']['apiKey']
        self._attrs['apiKeys']['edges'].append({'node': key})
        return key['name']
    except (requests.exceptions.HTTPError, AttributeError):
        return None