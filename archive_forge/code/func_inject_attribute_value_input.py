import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
def inject_attribute_value_input(self, params, model, **kwargs):
    """Injects DynamoDB serialization into parameter input"""
    self._transformer.transform(params, model.input_shape, self._serializer.serialize, 'AttributeValue')