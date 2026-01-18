from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def get_json_string(self, **kwargs) -> str:
    """Return string representation of JSON formatted table in the current state

        Keyword arguments are first interpreted as table formatting options, and
        then any unused keyword arguments are passed to json.dumps(). For
        example, get_json_string(header=False, indent=2) would use header as
        a PrettyTable formatting option (skip the header row) and indent as a
        json.dumps keyword argument.
        """
    import json
    options = self._get_options(kwargs)
    json_options: Any = {'indent': 4, 'separators': (',', ': '), 'sort_keys': True}
    json_options.update({key: value for key, value in kwargs.items() if key not in options})
    objects = []
    if options.get('header'):
        objects.append(self.field_names)
    for row in self._get_rows(options):
        objects.append(dict(zip(self._field_names, row)))
    return json.dumps(objects, **json_options)