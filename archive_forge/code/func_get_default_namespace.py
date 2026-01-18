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
def get_default_namespace(self):
    """
        Returns the default namespace.

        :return: :py:class:`~prov.identifier.Namespace`
        """
    return self._namespaces.get_default_namespace()