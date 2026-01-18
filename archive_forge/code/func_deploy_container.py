import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def deploy_container(self, name, image, parameters=None, start=True, **config):
    """
        Deploy a new container.

        http://docs.rancher.com/rancher/v1.2/en/api/api-resources/container/#create

        **The following is the Image format used for ``ContainerImage``**

        *For a ``imageuuid``*:

        - ``docker:<hostname>:<port>/<namespace>/<imagename>:<version>``

        *The following applies*:

        - ``id`` = ``<imagename>``
        - ``name`` = ``<imagename>``
        - ``path`` = ``<hostname>:<port>/<namespace>/<imagename>``
        - ``version`` = ``<version>``

        *Any extra configuration can also be passed i.e. "environment"*

        :param name: The desired name of the container. (required)
        :type name: ``str``

        :param image: The Image object to deploy. (required)
        :type image: :class:`libcloud.container.base.ContainerImage`

        :param parameters: Container Image parameters (unused)
        :type  parameters: ``str``

        :param start: Whether to start the container on creation(startOnCreate)
        :type start: ``bool``

        :rtype: :class:`Container`
        """
    payload = {'name': name, 'imageUuid': self._degen_image(image), 'startOnCreate': start}
    config.update(payload)
    data = json.dumps(config)
    result = self.connection.request('%s/containers' % self.baseuri, data=data, method='POST').object
    return self._to_container(result)