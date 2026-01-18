import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_additional_properties(self):
    """
        Means object with keys with values defined by definition.

        .. code-block:: python

            {
                'properties': {
                    'key': {'type': 'number'},
                }
                'additionalProperties': {'type': 'string'},
            }

        Valid object is containing key called 'key' and it's value any number and
        any other key with any string.
        """
    self.create_variable_is_dict()
    with self.l('if {variable}_is_dict:'):
        self.create_variable_keys()
        add_prop_definition = self._definition['additionalProperties']
        if add_prop_definition is True or add_prop_definition == {}:
            return
        if add_prop_definition:
            properties_keys = list(self._definition.get('properties', {}).keys())
            with self.l('for {variable}_key in {variable}_keys:'):
                with self.l('if {variable}_key not in {}:', properties_keys):
                    self.l('{variable}_value = {variable}.get({variable}_key)')
                    self.generate_func_code_block(add_prop_definition, '{}_value'.format(self._variable), '{}.{{{}_key}}'.format(self._variable_name, self._variable))
        else:
            with self.l('if {variable}_keys:'):
                self.exc('{name} must not contain "+str({variable}_keys)+" properties', rule='additionalProperties')