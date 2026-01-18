from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml_location_value
from googlecloudsdk.core.util import files
from ruamel import yaml
import six
def load_all_path(path, version=VERSION_1_1, round_trip=False):
    """Loads multiple YAML documents from the given file path.

  Args:
    path: str, A file path to open and read from.
    version: str, YAML version to use when parsing.
    round_trip: bool, True to use the RoundTripLoader which preserves ordering
      and line numbers.

  Raises:
    YAMLParseError: If the data could not be parsed.
    FileLoadError: If the file could not be opened or read.

  Yields:
    The parsed YAML data.
  """
    try:
        with files.FileReader(path) as fp:
            for x in load_all(fp, file_hint=path, version=version, round_trip=round_trip):
                yield x
    except files.Error as e:
        raise FileLoadError(e, f=path)