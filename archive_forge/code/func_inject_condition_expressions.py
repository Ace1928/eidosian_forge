import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
def inject_condition_expressions(self, params, model, **kwargs):
    """Injects the condition expression transformation into the parameters

        This injection includes transformations for ConditionExpression shapes
        and KeyExpression shapes. It also handles any placeholder names and
        values that are generated when transforming the condition expressions.
        """
    self._condition_builder.reset()
    generated_names = {}
    generated_values = {}
    transformation = ConditionExpressionTransformation(self._condition_builder, placeholder_names=generated_names, placeholder_values=generated_values, is_key_condition=False)
    self._transformer.transform(params, model.input_shape, transformation, 'ConditionExpression')
    transformation = ConditionExpressionTransformation(self._condition_builder, placeholder_names=generated_names, placeholder_values=generated_values, is_key_condition=True)
    self._transformer.transform(params, model.input_shape, transformation, 'KeyExpression')
    expr_attr_names_input = 'ExpressionAttributeNames'
    expr_attr_values_input = 'ExpressionAttributeValues'
    if expr_attr_names_input in params:
        params[expr_attr_names_input].update(generated_names)
    elif generated_names:
        params[expr_attr_names_input] = generated_names
    if expr_attr_values_input in params:
        params[expr_attr_values_input].update(generated_values)
    elif generated_values:
        params[expr_attr_values_input] = generated_values