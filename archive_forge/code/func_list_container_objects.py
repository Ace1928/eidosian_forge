import errno
import hashlib
import os.path  # pylint: disable-msg=W0404
import warnings
from typing import Dict, List, Type, Iterator, Optional
from os.path import join as pjoin
import libcloud.utils.files
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.storage.types import ObjectDoesNotExistError
def list_container_objects(self, container, prefix=None, ex_prefix=None):
    """
        Return a list of objects for the given container.

        :param container: Container instance.
        :type container: :class:`libcloud.storage.base.Container`

        :param prefix: Filter objects starting with a prefix.
        :type  prefix: ``str``

        :param ex_prefix: (Deprecated.) Filter objects starting with a prefix.
        :type  ex_prefix: ``str``

        :return: A list of Object instances.
        :rtype: ``list`` of :class:`libcloud.storage.base.Object`
        """
    return list(self.iterate_container_objects(container, prefix=prefix, ex_prefix=ex_prefix))