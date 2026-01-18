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
def is_described_by(self, media_type=None, query_values=None, representation_values=None):
    """Returns true if this method fits the given constraints.

        :param media_type: The method must accept this media type as a
                           representation.

        :param query_values: These key-value pairs must be acceptable
                           as values for this method's query
                           parameters. This need not be a complete set
                           of parameters acceptable to the method.

        :param representation_values: These key-value pairs must be
                           acceptable as values for this method's
                           representation parameters. Again, this need
                           not be a complete set of parameters
                           acceptable to the method.
        """
    representation = None
    if media_type is not None:
        representation = self.request.get_representation_definition(media_type)
        if representation is None:
            return False
    if query_values is not None and len(query_values) > 0:
        request = self.request
        if request is None:
            return False
        try:
            request.validate_param_values(request.query_params, query_values, False)
        except ValueError:
            return False
    if representation_values is None or len(representation_values) == 0:
        return True
    if representation is not None:
        return representation.is_described_by(representation_values)
    for representation in self.request.representations:
        try:
            representation.validate_param_values(representation.params(self.resource), representation_values, False)
            return True
        except ValueError:
            pass
    return False