from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
@property
def global_restriction(self):
    """The global restriction string or None if not a global restriction.

    Terms in a fiter expression are sometimes called "restrictions" because
    they restrict or constrain values.  A regular restriction is of the form
    "attribute<op>operand".  A "global restriction" is a term that has no
    attribute or <op>.  It is a bare string that is matched against every
    attribute value in the resource object being filtered.  The global
    restriction matches if any of those values contains the string using case
    insensitive string match.

    Returns:
      The global restriction string or None if not a global restriction.
    """
    if len(self._transforms) != 1 or self._transforms[0].name != resource_projection_spec.GLOBAL_RESTRICTION_NAME:
        return None
    return self._transforms[0].args[0]