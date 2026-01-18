from monascaclient.common import monasca_manager
def list_measurements(self, **kwargs):
    """Get a list of measurements based on metric definition filters."""
    return self._list('/measurements', 'dimensions', **kwargs)