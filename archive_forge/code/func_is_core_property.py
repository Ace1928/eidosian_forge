import copy
import json
import jsonpatch
import warlock.model as warlock
def is_core_property(self, property_name):
    """Check if a property with a given name is known to the schema.

        Determines if it is either a base property or a custom one
        registered in schema-image.json file

        :param property_name: name of the property
        :returns: True if the property is known, False otherwise
        """
    return self._check_property(property_name, True)