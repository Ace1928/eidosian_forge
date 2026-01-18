from __future__ import absolute_import, division, print_function
def get_az(module, fusion, availability_zone_name=None):
    """Get Availability Zone or None"""
    az_api_instance = purefusion.AvailabilityZonesApi(fusion)
    try:
        if availability_zone_name is None:
            availability_zone_name = module.params['availability_zone']
        return az_api_instance.get_availability_zone(region_name=module.params['region'], availability_zone_name=availability_zone_name)
    except purefusion.rest.ApiException:
        return None