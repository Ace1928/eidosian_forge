import abc
import collections
import copy
import functools
import hashlib
from stevedore import extension
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine import template_files
from heat.objects import raw_template as template_object
@classmethod
def create_empty_template(cls, version=('heat_template_version', '2015-04-30'), from_template=None):
    """Create an empty template.

        Creates a new empty template with given version. If version is
        not provided, a new empty HOT template of version "2015-04-30"
        is returned.

        :param version: A tuple containing version header of the template
                        version key and value,
                        e.g. ``('heat_template_version', '2015-04-30')``
        :returns: A new empty template.
        """
    if from_template:
        tmpl = copy.deepcopy(from_template)
        tmpl.remove_all_resources()
        return tmpl
    else:
        tmpl = {version[0]: version[1]}
        return cls(tmpl)