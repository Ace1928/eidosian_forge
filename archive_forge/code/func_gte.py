from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def gte(self, value):
    """Creates a condition where the attribute is greater than or equal to
           the value.

        :param value: The value that the attribute is greater than or equal to.
        """
    return GreaterThanEquals(self, value)