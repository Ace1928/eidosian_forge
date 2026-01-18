import os
import keras_tuner
import pytest
import autokeras as ak
from autokeras import graph as graph_module
def test_input_output_disconnect():
    input_node1 = ak.Input()
    output_node = input_node1
    _ = ak.DenseBlock()(output_node)
    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)
    with pytest.raises(ValueError) as info:
        graph_module.Graph(inputs=input_node1, outputs=output_node)
    assert 'Inputs and outputs not connected.' in str(info.value)