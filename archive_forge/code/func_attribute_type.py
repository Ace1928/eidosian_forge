from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def attribute_type(self, value):
    """Creates a condition for the attribute type.

        :param value: The type of the attribute.
        """
    return AttributeType(self, value)