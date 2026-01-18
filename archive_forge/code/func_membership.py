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
def membership(self, collection, entity):
    """
        Creates a new membership record for an entity to a collection.

        :param collection: Collection the entity is to be added to.
        :param entity: Entity to be added to the collection.
        """
    return self.new_record(PROV_MEMBERSHIP, None, {PROV_ATTR_COLLECTION: collection, PROV_ATTR_ENTITY: entity})