import os
import keras_tuner
import pytest
import autokeras as ak
from autokeras import graph as graph_module
def test_graph_can_init_with_one_missing_output():
    input_node = ak.ImageInput()
    output_node = ak.ConvBlock()(input_node)
    output_node = ak.RegressionHead()(output_node)
    ak.ClassificationHead()(output_node)
    graph_module.Graph(input_node, output_node)