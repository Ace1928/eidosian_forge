from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import units
from taskflow import task
from glance.common import exception as glance_exception
from glance.i18n import _LW
def get_glance_endpoint(context, region, interface):
    """Return glance endpoint depending the input tasks

    """
    catalog = context.service_catalog
    for service in catalog:
        if service['type'] == 'image':
            for endpoint in service['endpoints']:
                if endpoint['region'].lower() == region.lower():
                    return endpoint.get('%sURL' % interface)
    raise glance_exception.GlanceEndpointNotFound(region=region, interface=interface)