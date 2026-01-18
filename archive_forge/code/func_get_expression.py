from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def get_expression(self):
    return {'format': self.expression_format, 'operator': self.expression_operator, 'values': self._values}