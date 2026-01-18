import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
def copy_dynamodb_params(params, **kwargs):
    return copy.deepcopy(params)