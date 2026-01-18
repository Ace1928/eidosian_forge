from collections import defaultdict
from copy import deepcopy
import datetime
import io
import itertools
import logging
import os
import shutil
import tempfile
from urllib.parse import urlparse
import dateutil.parser
from prov import Error, serializers
from prov.constants import *
from prov.identifier import Identifier, QualifiedName, Namespace
def get_anonymous_identifier(self, local_prefix='id'):
    """
        Returns an anonymous identifier (without a namespace prefix).

        :param local_prefix: Optional local namespace prefix as a string
            (default: 'id').
        :return: :py:class:`~prov.identifier.Identifier`
        """
    self._anon_id_count += 1
    return Identifier('_:%s%d' % (local_prefix, self._anon_id_count))