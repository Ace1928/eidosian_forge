import logging
from oslo_utils import timeutils
from suds import sudsobject
def build_object_spec(client_factory, root_folder, traversal_specs):
    """Builds the object spec.

    :param client_factory: factory to get API input specs
    :param root_folder: root folder reference; the starting point of traversal
    :param traversal_specs: filter specs required for traversal
    :returns: object spec
    """
    object_spec = client_factory.create('ns0:ObjectSpec')
    object_spec.obj = root_folder
    object_spec.skip = False
    object_spec.selectSet = traversal_specs
    return object_spec