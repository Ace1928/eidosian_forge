from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import pkg_resources
import jsonschema
@staticmethod
def resolve_remote(ref):
    """pkg_resources $ref override -- schema_dir closure needed here."""
    path = os.path.join(schema_dir, ref)
    data = pkg_resources.GetResourceFromFile(path)
    try:
        schema = yaml.load(data)
    except Exception as e:
        raise InvalidSchemaError(e)
    self.ValidateSchemaVersion(schema, path)
    return schema