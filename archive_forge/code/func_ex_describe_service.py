from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def ex_describe_service(self, service_arn):
    """
        Get the details of a service

        :param  cluster: The hosting cluster
        :type   cluster: :class:`libcloud.container.base.ContainerCluster`

        :param  service_arn: The service ARN to describe
        :type   service_arn: ``str``

        :return: The service object
        :rtype: ``object``
        """
    request = {'services': [service_arn]}
    response = self.connection.request(ROOT, method='POST', data=json.dumps(request), headers=self._get_headers('DescribeServices')).object
    return response['services'][0]