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
def communication(self, informed, informant, identifier=None, other_attributes=None):
    """
        Creates a new communication record for an entity.

        :param informed: The informed activity (relationship destination).
        :param informant: The informing activity (relationship source).
        :param identifier: Identifier for new communication record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
    return self.new_record(PROV_COMMUNICATION, identifier, {PROV_ATTR_INFORMED: informed, PROV_ATTR_INFORMANT: informant}, other_attributes)