import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
def inject_attribute_value_output(self, parsed, model, **kwargs):
    """Injects DynamoDB deserialization into responses"""
    if model.output_shape is not None:
        self._transformer.transform(parsed, model.output_shape, self._deserializer.deserialize, 'AttributeValue')