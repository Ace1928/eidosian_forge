from monascaclient.common import monasca_manager
def list_names(self, **kwargs):
    """Get a list of metric names."""
    return self._list('/names', 'dimensions', **kwargs)