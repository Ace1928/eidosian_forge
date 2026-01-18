from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def ex_create_service(self, name, cluster, task_definition, desired_count=1):
    """
        Runs and maintains a desired number of tasks from a specified
        task definition. If the number of tasks running in a service
        drops below desired_count, Amazon ECS spawns another
        instantiation of the task in the specified cluster.

        :param  name: the name of the service
        :type   name: ``str``

        :param  cluster: The cluster to run the service on
        :type   cluster: :class:`libcloud.container.base.ContainerCluster`

        :param  task_definition: The task definition name or ARN for the
            service
        :type   task_definition: ``str``

        :param  desired_count: The desired number of tasks to be running
            at any one time
        :type   desired_count: ``int``

        :rtype: ``object`` The service object
        """
    request = {'serviceName': name, 'taskDefinition': task_definition, 'desiredCount': desired_count, 'cluster': cluster.id}
    response = self.connection.request(ROOT, method='POST', data=json.dumps(request), headers=self._get_headers('CreateService')).object
    return response['service']