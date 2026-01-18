import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('LogisticRegressionOutput')
def convert_logistic_regression_output(node, **kwargs):
    """Map MXNet's SoftmaxOutput operator attributes to onnx's Softmax operator
    and return the created node.
    """
    name = node['name']
    input1 = kwargs['outputs_lookup'][node['inputs'][0][0]][node['inputs'][0][1]].name
    sigmoid_node = onnx.helper.make_node('Sigmoid', [input1], [name], name=name)
    return [sigmoid_node]