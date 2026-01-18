import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_deploy_stack(self, name, description=None, docker_compose=None, environment=None, external_id=None, rancher_compose=None, start=True):
    """
        Deploy a new stack.

        http://docs.rancher.com/rancher/v1.2/en/api/api-resources/environment/#create

        :param name: The desired name of the stack. (required)
        :type name: ``str``

        :param description: A desired description for the stack.
        :type description: ``str``

        :param docker_compose: The Docker Compose configuration to use.
        :type docker_compose: ``str``

        :param environment: Environment K/V specific to this stack.
        :type environment: ``dict``

        :param external_id: The externalId of the stack.
        :type external_id: ``str``

        :param rancher_compose: The Rancher Compose configuration for this env.
        :type rancher_compose: ``str``

        :param start: Whether to start this stack on creation.
        :type start: ``bool``

        :return: The newly created stack.
        :rtype: ``dict``
        """
    payload = {'description': description, 'dockerCompose': docker_compose, 'environment': environment, 'externalId': external_id, 'name': name, 'rancherCompose': rancher_compose, 'startOnCreate': start}
    data = json.dumps({k: v for k, v in payload.items() if v is not None})
    result = self.connection.request('%s/environments' % self.baseuri, data=data, method='POST').object
    return result