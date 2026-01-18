from monascaclient.common import monasca_manager
def list_dimension_values(self, **kwargs):
    """Get a list of metric dimension values."""
    return self._list('/dimensions/names/values', **kwargs)