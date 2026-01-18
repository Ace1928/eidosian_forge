from monascaclient.common import monasca_manager
def list_statistics(self, **kwargs):
    """Get a list of measurement statistics based on metric def filters."""
    return self._list('/statistics', 'dimensions', **kwargs)