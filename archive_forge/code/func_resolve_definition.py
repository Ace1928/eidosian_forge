import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
def resolve_definition(self):
    """Return the definition of this object, wherever it is.

        Resource is a good example. A WADL <resource> tag
        may contain a large number of nested tags describing a
        resource, or it may just contain a 'type' attribute that
        references a <resource_type> which contains those same
        tags. Resource.resolve_definition() will return the original
        Resource object in the first case, and a
        ResourceType object in the second case.
        """
    if self._definition is not None:
        return self._definition
    object_url = self._get_definition_url()
    if object_url is None:
        self._definition = self
        return self
    xml_id = self.application.lookup_xml_id(object_url)
    definition = self._definition_factory(xml_id)
    if definition is None:
        raise KeyError('No such XML ID: "%s"' % object_url)
    self._definition = definition
    return definition