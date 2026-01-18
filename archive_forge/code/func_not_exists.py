from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def not_exists(self):
    """Creates a condition where the attribute does not exist."""
    return AttributeNotExists(self)