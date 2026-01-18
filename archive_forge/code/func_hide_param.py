import re
from collections import namedtuple
def hide_param(self, event_name, section, **kwargs):
    if event_name in self._example_events:
        section = section.get_section('structure-value')
    elif event_name not in self._params_events:
        return
    if self._parameter_name in section.available_sections:
        section.delete_section(self._parameter_name)