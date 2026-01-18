import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def deregister_container_instance(self, container_instance, cluster=None, force=None):
    """
        Deregisters an Amazon ECS container instance from the
        specified cluster. This instance will no longer be available
        to run tasks.

        :type cluster: string
        :param cluster: The short name or full Amazon Resource Name (ARN) of
            the cluster that hosts the container instance you want to
            deregister. If you do not specify a cluster, the default cluster is
            assumed.

        :type container_instance: string
        :param container_instance: The container instance UUID or full Amazon
            Resource Name (ARN) of the container instance you want to
            deregister. The ARN contains the `arn:aws:ecs` namespace, followed
            by the region of the container instance, the AWS account ID of the
            container instance owner, the `container-instance` namespace, and
            then the container instance UUID. For example, arn:aws:ecs: region
            : aws_account_id :container-instance/ container_instance_UUID .

        :type force: boolean
        :param force: Force the deregistration of the container instance. You
            can use the `force` parameter if you have several tasks running on
            a container instance and you don't want to run `StopTask` for each
            task before deregistering the container instance.

        """
    params = {'containerInstance': container_instance}
    if cluster is not None:
        params['cluster'] = cluster
    if force is not None:
        params['force'] = str(force).lower()
    return self._make_request(action='DeregisterContainerInstance', verb='POST', path='/', params=params)