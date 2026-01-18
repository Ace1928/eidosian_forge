def map_ie(self, ie):
    """Fix the references to old file ids in an inventory entry.

        :param ie: Inventory entry to map
        :return: New inventory entry
        """
    new_ie = ie.copy()
    new_ie.file_id = self.new_id(new_ie.file_id)
    new_ie.parent_id = self.new_id(new_ie.parent_id)
    return new_ie