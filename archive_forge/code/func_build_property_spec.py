import logging
from oslo_utils import timeutils
from suds import sudsobject
def build_property_spec(client_factory, type_='VirtualMachine', properties_to_collect=None, all_properties=False):
    """Builds the property spec.

    :param client_factory: factory to get API input specs
    :param type_: type of the managed object
    :param properties_to_collect: names of the managed object properties to be
                                  collected while traversal filtering
    :param all_properties: whether all properties of the managed object need
                           to be collected
    :returns: property spec
    """
    if not properties_to_collect:
        properties_to_collect = ['name']
    property_spec = client_factory.create('ns0:PropertySpec')
    property_spec.all = all_properties
    property_spec.pathSet = properties_to_collect
    property_spec.type = type_
    return property_spec