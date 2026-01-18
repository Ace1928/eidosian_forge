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
def alternateOf(self, alternate2):
    """
        Creates a new alternate record between this and another entity.

        :param alternate2: Entity or a string identifier for the second entity.
        """
    self._bundle.alternate(self, alternate2)
    return self