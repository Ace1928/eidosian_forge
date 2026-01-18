from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS
from mlflow.transformers.hub_utils import get_latest_commit_for_repo
from mlflow.transformers.peft import _PEFT_ADAPTOR_DIR_NAME, get_peft_base_model, is_peft_model
from mlflow.transformers.torch_utils import _extract_torch_dtype_if_set
def build_flavor_config(pipeline: transformers.Pipeline, processor=None, torch_dtype=None, save_pretrained=True) -> Dict[str, Any]:
    """
    Generates the base flavor metadata needed for reconstructing a pipeline from saved
    components. This is important because the ``Pipeline`` class does not have a loader
    functionality. The serialization of a Pipeline saves the model, configurations, and
    metadata for ``FeatureExtractor``s, ``Processor``s, and ``Tokenizer``s exclusively.
    This function extracts key information from the submitted model object so that the precise
    instance types can be loaded correctly.

    Args:
        pipeline: Transformer pipeline to generate the flavor configuration for.
        processor: Optional processor instance to save alongside the pipeline.
        save_pretrained: Whether to save the pipeline and components weights to local disk.

    Returns:
        A dictionary containing the flavor configuration for the pipeline and its components,
        i.e. the configurations stored in "transformers" key in the MLModel YAML file.
    """
    flavor_conf = _generate_base_config(pipeline, torch_dtype=torch_dtype)
    if is_peft_model(pipeline.model):
        flavor_conf[FlavorKey.PEFT] = _PEFT_ADAPTOR_DIR_NAME
        model = get_peft_base_model(pipeline.model)
    else:
        model = pipeline.model
    flavor_conf.update(_get_model_config(model, save_pretrained))
    components = _get_components_from_pipeline(pipeline, processor)
    for key, instance in components.items():
        flavor_conf.update(_get_component_config(instance, key, save_pretrained, default_repo=model.name_or_path))
    components.pop(FlavorKey.PROCESSOR, None)
    flavor_conf[FlavorKey.COMPONENTS] = list(components.keys())
    return flavor_conf