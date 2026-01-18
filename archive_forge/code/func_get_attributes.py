import logging
from botocore import xform_name
def get_attributes(self, shape):
    """
        Get a dictionary of attribute names to original name and shape
        models that represent the attributes of this resource. Looks
        like the following:

            {
                'some_name': ('SomeName', <Shape...>)
            }

        :type shape: botocore.model.Shape
        :param shape: The underlying shape for this resource.
        :rtype: dict
        :return: Mapping of resource attributes.
        """
    attributes = {}
    identifier_names = [i.name for i in self.identifiers]
    for name, member in shape.members.items():
        snake_cased = xform_name(name)
        if snake_cased in identifier_names:
            continue
        snake_cased = self._get_name('attribute', snake_cased, snake_case=False)
        attributes[snake_cased] = (name, member)
    return attributes