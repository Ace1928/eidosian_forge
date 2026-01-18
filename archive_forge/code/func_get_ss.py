from __future__ import absolute_import, division, print_function
def get_ss(module, fusion, storage_service_name=None):
    """Return Storage Service or None"""
    ss_api_instance = purefusion.StorageServicesApi(fusion)
    try:
        if storage_service_name is None:
            storage_service_name = module.params['storage_service']
        return ss_api_instance.get_storage_service(storage_service_name=storage_service_name)
    except purefusion.rest.ApiException:
        return None