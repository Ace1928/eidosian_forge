import os
import keras_tuner
import pytest
import autokeras as ak
from autokeras import graph as graph_module
def test_cat_to_num_with_img_input_error():
    input_node = ak.ImageInput()
    output_node = ak.CategoricalToNumerical()(input_node)
    with pytest.raises(TypeError) as info:
        graph_module.Graph(input_node, outputs=output_node).compile()
    assert 'CategoricalToNumerical can only be used' in str(info.value)