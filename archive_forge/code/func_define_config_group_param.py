import inspect
import re
import six
def define_config_group_param(self, group, param, type, description=None, writable=True):
    """
        Helper to define configuration group parameters.
        @param group: The configuration group to add the parameter to.
        @type group: str
        @param param: The new parameter name.
        @type param: str
        @param description: Optional description string.
        @type description: str
        @param writable: Whether or not this would be a rw or ro parameter.
        @type writable: bool
        """
    if group not in self._configuration_groups:
        self._configuration_groups[group] = {}
    if description is None:
        description = 'The %s %s parameter.' % (param, group)
    self.get_type_method(type)
    self.get_group_getter(group)
    if writable:
        self.get_group_setter(group)
    self._configuration_groups[group][param] = [type, description, writable]