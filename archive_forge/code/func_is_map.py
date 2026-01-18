@property
def is_map(self):
    """Return True if the descriptor is a map entry, False otherwise."""
    desc = self._descriptor.DESCRIPTOR
    return desc.has_options and desc.GetOptions().map_entry