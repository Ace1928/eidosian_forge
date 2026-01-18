import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def set_common_instance_metadata(self, metadata=None, force=False):
    """
        Set common instance metadata for the project. Common uses
        are for setting 'sshKeys', or setting a project-wide
        'startup-script' for all nodes (instances).  Passing in
        ``None`` for the 'metadata' parameter will clear out all common
        instance metadata *except* for 'sshKeys'. If you also want to
        update 'sshKeys', set the 'force' parameter to ``True``.

        :param  metadata: Dictionary of metadata. Can be either a standard
                          python dictionary, or the format expected by
                          GCE (e.g. {'items': [{'key': k1, 'value': v1}, ...}]
        :type   metadata: ``dict`` or ``None``

        :param  force: Force update of 'sshKeys'. If force is ``False`` (the
                       default), existing sshKeys will be retained. Setting
                       force to ``True`` will either replace sshKeys if a new
                       a new value is supplied, or deleted if no new value
                       is supplied.
        :type   force: ``bool``

        :return: True if successful
        :rtype:  ``bool``
        """
    return self.driver.ex_set_common_instance_metadata(metadata=metadata, force=force)