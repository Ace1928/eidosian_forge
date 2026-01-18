import boto.vendored.regions.regions as _regions
def resolve_hostname(self, service_name, region_name):
    """Resolve the hostname for a service in a particular region.

        :type service_name: str
        :param service_name: The service to look up.

        :type region_name: str
        :param region_name: The region to find the endpoint for.

        :return: The hostname for the given service in the given region.
        """
    endpoint = self._resolver.construct_endpoint(service_name, region_name)
    if endpoint is None:
        return None
    return endpoint.get('sslCommonName', endpoint['hostname'])