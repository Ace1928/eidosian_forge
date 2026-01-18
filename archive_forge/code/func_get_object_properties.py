import logging
from oslo_utils import timeutils
from suds import sudsobject
def get_object_properties(vim, moref, properties_to_collect, skip_op_id=False):
    """Get properties of the given managed object.

    :param vim: Vim object
    :param moref: managed object reference
    :param properties_to_collect: names of the managed object properties to be
                                  collected
    :param skip_op_id: whether to skip putting opID in the request
    :returns: properties of the given managed object
    :raises: VimException, VimFaultException, VimAttributeException,
             VimSessionOverLoadException, VimConnectionException
    """
    if moref is None:
        return None
    client_factory = vim.client.factory
    all_properties = properties_to_collect is None or len(properties_to_collect) == 0
    property_spec = build_property_spec(client_factory, type_=get_moref_type(moref), properties_to_collect=properties_to_collect, all_properties=all_properties)
    object_spec = build_object_spec(client_factory, moref, [])
    property_filter_spec = build_property_filter_spec(client_factory, [property_spec], [object_spec])
    options = client_factory.create('ns0:RetrieveOptions')
    options.maxObjects = 1
    retrieve_result = vim.RetrievePropertiesEx(vim.service_content.propertyCollector, specSet=[property_filter_spec], options=options, skip_op_id=skip_op_id)
    cancel_retrieval(vim, retrieve_result)
    return retrieve_result.objects