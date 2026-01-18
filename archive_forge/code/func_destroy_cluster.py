from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def destroy_cluster(self, cluster):
    """
        Delete a cluster

        :return: ``True`` if the destroy was successful, otherwise ``False``.
        :rtype: ``bool``
        """
    request = {'cluster': cluster.id}
    data = self.connection.request(ROOT, method='POST', data=json.dumps(request), headers=self._get_headers('DeleteCluster')).object
    return data['cluster']['status'] == 'INACTIVE'