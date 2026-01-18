from copy import deepcopy
def partial_save(self):
    """
        Saves only the changed data to DynamoDB.

        Extremely useful for high-volume/high-write data sets, this allows
        you to update only a handful of fields rather than having to push
        entire items. This prevents many accidental overwrite situations as
        well as saves on the amount of data to transfer over the wire.

        Returns ``True`` on success, ``False`` if no save was performed or
        the write failed.

        Example::

            >>> user['last_name'] = 'Doh!'
            # Only the last name field will be sent to DynamoDB.
            >>> user.partial_save()

        """
    key = self.get_keys()
    final_data, fields = self.prepare_partial()
    if not final_data:
        return False
    for fieldname, value in key.items():
        if fieldname in final_data:
            del final_data[fieldname]
            try:
                fields.remove(fieldname)
            except KeyError:
                pass
    expects = self.build_expects(fields=fields)
    returned = self.table._update_item(key, final_data, expects=expects)
    self.mark_clean()
    return returned