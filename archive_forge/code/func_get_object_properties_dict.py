import logging
from oslo_utils import timeutils
from suds import sudsobject
def get_object_properties_dict(vim, moref, properties_to_collect):
    """Get properties of the given managed object as a dict.

    :param vim: Vim object
    :param moref: managed object reference
    :param properties_to_collect: names of the managed object properties to be
                                  collected
    :returns: a dict of properties of the given managed object
    :raises: VimException, VimFaultException, VimAttributeException,
             VimSessionOverLoadException, VimConnectionException
    """
    obj_contents = get_object_properties(vim, moref, properties_to_collect)
    if obj_contents is None:
        return {}
    property_dict = {}
    if hasattr(obj_contents[0], 'propSet'):
        dynamic_properties = obj_contents[0].propSet
        if dynamic_properties:
            for prop in dynamic_properties:
                property_dict[prop.name] = prop.val
    if hasattr(obj_contents[0], 'missingSet'):
        for m in obj_contents[0].missingSet:
            LOG.warning('Unable to retrieve value for %(path)s Reason: %(reason)s', {'path': m.path, 'reason': m.fault.localizedMessage})
    return property_dict