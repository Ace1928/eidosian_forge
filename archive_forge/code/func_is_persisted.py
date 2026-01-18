@property
def is_persisted(self):
    """Determine if the resource for this event payload is persisted.

        :returns: True if this payload's resource is persisted, otherwise
            False.
        """
    return self.resource_id is not None and self.has_states