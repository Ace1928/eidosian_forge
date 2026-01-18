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
def all_param_schemata(self, files):
    schema = {}
    files = files if files is not None else {}
    for f in files.values():
        try:
            data = template_format.parse(f)
        except ValueError:
            continue
        else:
            sub_tmpl = Template(data)
            schema.update(sub_tmpl.param_schemata())
    schema.update(self.param_schemata())
    return schema