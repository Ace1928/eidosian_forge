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
def set_default_namespace(self, uri):
    """
        Sets the default namespace through a given URI.

        :param uri: Namespace URI.
        """
    self._namespaces.set_default_namespace(uri)