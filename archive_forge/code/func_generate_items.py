import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_items(self):
    """
        Means array is valid only when all items are valid by this definition.

        .. code-block:: python

            {
                'items': [
                    {'type': 'integer'},
                    {'type': 'string'},
                ],
            }

        Valid arrays are those with integers or strings, nothing else.

        Since draft 06 definition can be also boolean. True means nothing, False
        means everything is invalid.
        """
    items_definition = self._definition['items']
    if items_definition is True:
        return
    self.create_variable_is_list()
    with self.l('if {variable}_is_list:'):
        self.create_variable_with_length()
        if items_definition is False:
            with self.l('if {variable}:'):
                self.exc('{name} must not be there', rule='items')
        elif isinstance(items_definition, list):
            for idx, item_definition in enumerate(items_definition):
                with self.l('if {variable}_len > {}:', idx):
                    self.l('{variable}__{0} = {variable}[{0}]', idx)
                    self.generate_func_code_block(item_definition, '{}__{}'.format(self._variable, idx), '{}[{}]'.format(self._variable_name, idx))
                if self._use_default and isinstance(item_definition, dict) and ('default' in item_definition):
                    self.l('else: {variable}.append({})', repr(item_definition['default']))
            if 'additionalItems' in self._definition:
                if self._definition['additionalItems'] is False:
                    with self.l('if {variable}_len > {}:', len(items_definition)):
                        self.exc('{name} must contain only specified items', rule='items')
                else:
                    with self.l('for {variable}_x, {variable}_item in enumerate({variable}[{0}:], {0}):', len(items_definition)):
                        count = self.generate_func_code_block(self._definition['additionalItems'], '{}_item'.format(self._variable), '{}[{{{}_x}}]'.format(self._variable_name, self._variable))
                        if count == 0:
                            self.l('pass')
        elif items_definition:
            with self.l('for {variable}_x, {variable}_item in enumerate({variable}):'):
                count = self.generate_func_code_block(items_definition, '{}_item'.format(self._variable), '{}[{{{}_x}}]'.format(self._variable_name, self._variable))
                if count == 0:
                    self.l('pass')