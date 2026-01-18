import os
import re
import shlex
import base64
import datetime
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_search_images(self, term):
    """Search for an image on Docker.io.
        Returns a list of ContainerImage objects

        >>> images = conn.ex_search_images(term='mistio')
        >>> images
        [<ContainerImage: id=rolikeusch/docker-mistio...>,
         <ContainerImage: id=mist/mistio, name=mist/mistio,
             driver=Docker  ...>]

         :param term: The search term
         :type  term: ``str``

         :rtype: ``list`` of :class:`libcloud.container.base.ContainerImage`
        """
    term = term.replace(' ', '+')
    result = self.connection.request('/v{}/images/search?term={}'.format(self.version, term)).object
    images = []
    for image in result:
        name = image.get('name')
        images.append(ContainerImage(id=name, path=name, version=None, name=name, driver=self.connection.driver, extra={'description': image.get('description'), 'is_official': image.get('is_official'), 'is_trusted': image.get('is_trusted'), 'star_count': image.get('star_count')}))
    return images