import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_deploy_service(self, name, image, environment_id, start=True, assign_service_ip_address=None, service_description=None, external_id=None, metadata=None, retain_ip=None, scale=None, scale_policy=None, secondary_launch_configs=None, selector_container=None, selector_link=None, vip=None, **launch_conf):
    """
        Deploy a Rancher Service under a stack.

        http://docs.rancher.com/rancher/v1.2/en/api/api-resources/service/#create

        *Any further configuration passed applies to the ``launchConfig``*

        :param name: The desired name of the service. (required)
        :type name: ``str``

        :param image: The Image object to deploy. (required)
        :type image: :class:`libcloud.container.base.ContainerImage`

        :param environment_id: The stack ID this service is tied to. (required)
        :type environment_id: ``str``

        :param start: Whether to start the service on creation.
        :type start: ``bool``

        :param assign_service_ip_address: The IP address to assign the service.
        :type assign_service_ip_address: ``bool``

        :param service_description: The service description.
        :type service_description: ``str``

        :param external_id: The externalId for this service.
        :type external_id: ``str``

        :param metadata: K/V Metadata for this service.
        :type metadata: ``dict``

        :param retain_ip: Whether this service should retain its IP.
        :type retain_ip: ``bool``

        :param scale: The scale of containers in this service.
        :type scale: ``int``

        :param scale_policy: The scaling policy for this service.
        :type scale_policy: ``dict``

        :param secondary_launch_configs: Secondary container launch configs.
        :type secondary_launch_configs: ``list``

        :param selector_container: The selectorContainer for this service.
        :type selector_container: ``str``

        :param selector_link: The selectorLink for this service.
        :type selector_link: ``type``

        :param vip: The VIP to assign to this service.
        :type vip: ``str``

        :return: The newly created service.
        :rtype: ``dict``
        """
    launch_conf['imageUuid'] = (self._degen_image(image),)
    service_payload = {'assignServiceIpAddress': assign_service_ip_address, 'description': service_description, 'environmentId': environment_id, 'externalId': external_id, 'launchConfig': launch_conf, 'metadata': metadata, 'name': name, 'retainIp': retain_ip, 'scale': scale, 'scalePolicy': scale_policy, 'secondary_launch_configs': secondary_launch_configs, 'selectorContainer': selector_container, 'selectorLink': selector_link, 'startOnCreate': start, 'vip': vip}
    data = json.dumps({k: v for k, v in service_payload.items() if v is not None})
    result = self.connection.request('%s/services' % self.baseuri, data=data, method='POST').object
    return result