from typing import Callable, Optional, Union
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
@DeveloperAPI
def input_to_output_specs(input_specs: SpecDict, num_input_feature_dims: int, output_key: str, output_feature_spec: TensorSpec) -> SpecDict:
    """Convert an input spec to an output spec, based on a module.

    Drops the feature dimension(s) from an input_specs, replacing them with
    output_feature_spec dimension(s).

    Examples:
        input_to_output_specs(
            input_specs=SpecDict({
                "bork": "batch, time, feature0",
                "dork": "batch, time, feature1"
                }, feature0=2, feature1=3
            ),
            num_input_feature_dims=1,
            output_key="outer_product",
            output_feature_spec=TensorSpec("row, col", row=2, col=3)
        )

        will return:
        SpecDict({"outer_product": "batch, time, row, col", row=2, col=3})

        input_to_output_specs(
            input_specs=SpecDict({
                "bork": "batch, time, h, w, c",
                }, h=32, w=32, c=3,
            ),
            num_input_feature_dims=3,
            output_key="latent_image_representation",
            output_feature_spec=TensorSpec("feature", feature=128)
        )

        will return:
        SpecDict({"latent_image_representation": "batch, time, feature"}, feature=128)


    Args:
        input_specs: SpecDict describing input to a specified module
        num_input_dims: How many feature dimensions the module will process. E.g.
            a linear layer will only process the last dimension (1), while a CNN
            might process the last two dimensions (2)
        output_key: The key in the output spec we will write the resulting shape to
        output_feature_spec: A spec denoting the feature dimensions output by a
            specified module

    Returns:
        A SpecDict based on the input_specs, with the trailing dimensions replaced
            by the output_feature_spec

    """
    assert num_input_feature_dims >= 1, 'Must specify at least one feature dim'
    num_dims = [len(v.shape) != len for v in input_specs.values()]
    assert all((nd == num_dims[0] for nd in num_dims)), 'All specs in input_specs must all have the same number of dimensions'
    key = list(input_specs.keys())[0]
    batch_spec = input_specs[key].rdrop(num_input_feature_dims)
    full_spec = batch_spec.append(output_feature_spec)
    return SpecDict({output_key: full_spec})