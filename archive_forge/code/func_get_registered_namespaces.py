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
def get_registered_namespaces(self):
    """
        Returns all registered namespaces.

        :return: Iterable of :py:class:`~prov.identifier.Namespace`.
        """
    return self._namespaces.get_registered_namespaces()