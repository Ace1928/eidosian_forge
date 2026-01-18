from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import shlex
def parse_others(self, other_arg, key, value):
    """Gets metadata from an example string for non list-type/dict-type args.

    It updates the already existing ExampleCommandMetadata object of the example
    string with the metadata.

    Args:
      other_arg: The non list-type and non dict-type argument to collect
      metadata about.
      key: The name of the argument's attribute in the example string's
      namespace.
      value: The actual input representing the flag/argument in the example
      string (e.g --zone, --shielded-secure-boot).

    """
    if not isinstance(other_arg, bool):
        start_search = self.get_start_search(value)
        start_index = self.example_string.find(str(other_arg), start_search)
        if start_index == -1:
            other_arg = self.get_enum_value(other_arg, start_search)
        start_index = self.example_string.index(str(other_arg), start_search)
        arg_metadata = ArgumentMetadata(key, other_arg, key, start_index, start_index + len(str(other_arg)) - 1)
        self.example_metadata.add_arg_metadata(arg_metadata)