from octavia_lib.api.drivers import exceptions
def get_supported_availability_zone_metadata(self):
    """Returns a dict of supported availability zone metadata keys.

        The returned dictionary will include key/value pairs, 'name' and
        'description.'

        :returns: The availability zone metadata dictionary
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: The driver does not support AZs.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support getting the supported availability zone metadata.', operator_fault_string='This provider does not support getting the supported availability zone metadata.')