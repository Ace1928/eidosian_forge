from boto.dynamodb.exceptions import DynamoDBItemError
@property
def hash_key_name(self):
    return self._hash_key_name