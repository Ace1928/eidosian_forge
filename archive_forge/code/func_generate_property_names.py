import decimal
from .draft04 import CodeGeneratorDraft04, JSON_TYPE_TO_PYTHON_TYPE
from .exceptions import JsonSchemaDefinitionException
from .generator import enforce_list
def generate_property_names(self):
    """
        Means that keys of object must to follow this definition.

        .. code-block:: python

            {
                'propertyNames': {
                    'maxLength': 3,
                },
            }

        Valid keys of object for this definition are foo, bar, ... but not foobar for example.
        """
    property_names_definition = self._definition.get('propertyNames', {})
    if property_names_definition is True:
        pass
    elif property_names_definition is False:
        self.create_variable_keys()
        with self.l('if {variable}_keys:'):
            self.exc('{name} must not be there', rule='propertyNames')
    else:
        self.create_variable_is_dict()
        with self.l('if {variable}_is_dict:'):
            self.create_variable_with_length()
            with self.l('if {variable}_len != 0:'):
                self.l('{variable}_property_names = True')
                with self.l('for {variable}_key in {variable}:'):
                    with self.l('try:'):
                        self.generate_func_code_block(property_names_definition, '{}_key'.format(self._variable), self._variable_name, clear_variables=True)
                    with self.l('except JsonSchemaValueException:'):
                        self.l('{variable}_property_names = False')
                with self.l('if not {variable}_property_names:'):
                    self.exc('{name} must be named by propertyName definition', rule='propertyNames')