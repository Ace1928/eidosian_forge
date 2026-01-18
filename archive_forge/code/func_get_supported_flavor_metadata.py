from octavia_lib.api.drivers import exceptions
def get_supported_flavor_metadata(self):
    """Returns a dict of flavor metadata keys supported by this driver.

        The returned dictionary will include key/value pairs, 'name' and
        'description.'

        :returns: The flavor metadata dictionary
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: The driver does not support flavors.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support getting the supported flavor metadata.', operator_fault_string='This provider does not support getting the supported flavor metadata.')