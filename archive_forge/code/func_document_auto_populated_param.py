import re
from collections import namedtuple
def document_auto_populated_param(self, event_name, section, **kwargs):
    """Documents auto populated parameters

        It will remove any required marks for the parameter, remove the
        parameter from the example, and add a snippet about the parameter
        being autopopulated in the description.
        """
    if event_name.startswith('docs.request-params'):
        if self.name in section.available_sections:
            section = section.get_section(self.name)
            if 'is-required' in section.available_sections:
                section.delete_section('is-required')
            description_section = section.get_section('param-documentation')
            description_section.writeln(self.param_description)
    elif event_name.startswith('docs.request-example'):
        section = section.get_section('structure-value')
        if self.name in section.available_sections:
            section.delete_section(self.name)