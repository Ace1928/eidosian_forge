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
def mention(self, specificEntity, generalEntity, bundle):
    """
        Creates a new mention record for a specific from a general entity.

        :param specificEntity: Entity or a string identifier for the specific
            entity (relationship source).
        :param generalEntity: Entity or a string identifier for the general entity
            (relationship destination).
        :param bundle: XXX
        """
    return self.new_record(PROV_MENTION, None, {PROV_ATTR_SPECIFIC_ENTITY: specificEntity, PROV_ATTR_GENERAL_ENTITY: generalEntity, PROV_ATTR_BUNDLE: bundle})