import logging
from oslo_utils import timeutils
from suds import sudsobject
def build_property_filter_spec(client_factory, property_specs, object_specs):
    """Builds the property filter spec.

    :param client_factory: factory to get API input specs
    :param property_specs: property specs to be collected for filtered objects
    :param object_specs: object specs to identify objects to be filtered
    :returns: property filter spec
    """
    property_filter_spec = client_factory.create('ns0:PropertyFilterSpec')
    property_filter_spec.propSet = property_specs
    property_filter_spec.objectSet = object_specs
    return property_filter_spec