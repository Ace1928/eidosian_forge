from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def ex_list_service_arns(self, cluster=None):
    """
        List the services

        :param cluster: The cluster hosting the services
        :type  cluster: :class:`libcloud.container.base.ContainerCluster`

        :rtype: ``list`` of ``str``
        """
    request = {}
    if cluster is not None:
        request['cluster'] = cluster.id
    response = self.connection.request(ROOT, method='POST', data=json.dumps(request), headers=self._get_headers('ListServices')).object
    return response['serviceArns']