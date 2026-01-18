import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
def skip_template(condition=True, *args):
    """
    Raise SkipTemplate, which causes copydir to skip the template
    being processed.  If you pass in a condition, only raise if that
    condition is true (allows you to use this with string.Template)

    If you pass any additional arguments, they will be used to
    instantiate SkipTemplate (generally use like
    ``skip_template(license=='GPL', 'Skipping file; not using GPL')``)
    """
    if condition:
        raise SkipTemplate(*args)