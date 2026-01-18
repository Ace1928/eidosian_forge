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
def primary_source(self, generatedEntity, usedEntity, activity=None, generation=None, usage=None, identifier=None, other_attributes=None):
    """
        Creates a new primary source record for a generated entity from a used
        entity.

        :param generatedEntity: Entity or a string identifier for the generated
            entity (relationship source).
        :param usedEntity: Entity or a string identifier for the used entity
            (relationship destination).
        :param activity: Activity or string identifier of the activity involved in
            the primary source (default: None).
        :param generation: Optionally to state qualified primary source through a
            generation activity (default: None).
        :param usage: XXX (default: None).
        :param identifier: Identifier for new primary source record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
    record = self.derivation(generatedEntity, usedEntity, activity, generation, usage, identifier, other_attributes)
    record.add_asserted_type(PROV['PrimarySource'])
    return record