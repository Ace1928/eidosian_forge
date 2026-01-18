from boto.dynamodb.exceptions import DynamoDBItemError
def delete_attribute(self, attr_name, attr_value=None):
    """
        Queue the deletion of an attribute from an item in DynamoDB.
        This call will result in a UpdateItem request being issued
        with update action of DELETE when the save method is called.

        :type attr_name: str
        :param attr_name: Name of the attribute you want to alter.

        :type attr_value: set
        :param attr_value: A set of values to be removed from the attribute.
            This parameter is optional. If None, the whole attribute is
            removed from the item.
        """
    self._updates[attr_name] = ('DELETE', attr_value)