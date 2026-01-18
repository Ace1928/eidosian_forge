from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def ex_get_repository_token(self, repository_id):
    """
        Get the authorization token (12 hour expiry) for a repository

        :param  repository_id: The ID of the repository
        :type   repository_id: ``str``

        :return: A token for login
        :rtype: ``str``
        """
    request = {'RegistryIds': [repository_id]}
    response = self.ecr_connection.request(ROOT, method='POST', data=json.dumps(request), headers=self._get_ecr_headers('GetAuthorizationToken')).object
    return response['authorizationData'][0]['authorizationToken']