import os
import keras_tuner
import pytest
import autokeras as ak
from autokeras import graph as graph_module
def test_input_missing():
    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    output_node = ak.RegressionHead()(output_node)
    with pytest.raises(ValueError) as info:
        graph_module.Graph(inputs=input_node1, outputs=output_node)
    assert 'A required input is missing for HyperModel' in str(info.value)