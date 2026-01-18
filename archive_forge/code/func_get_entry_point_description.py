import textwrap
import os
import pkg_resources
from .command import Command, BadCommand
import fnmatch
import re
import traceback
from io import StringIO
import inspect
import types
def get_entry_point_description(self, ep, group):
    try:
        return self._safe_get_entry_point_description(ep, group)
    except Exception as e:
        out = StringIO()
        traceback.print_exc(file=out)
        return ErrorDescription(e, out.getvalue())