from copy import deepcopy
def prepare_full(self):
    """
        Runs through all fields & encodes them to be handed off to DynamoDB
        as part of an ``save`` (``put_item``) call.

        Largely internal.
        """
    final_data = {}
    for key, value in self._data.items():
        if not self._is_storable(value):
            continue
        final_data[key] = self._dynamizer.encode(value)
    return final_data