import logging
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE
from mlflow.transformers.flavor_config import FlavorKey, get_peft_base_model, is_peft_model
def save_pipeline_pretrained_weights(path, pipeline, flavor_conf, processor=None):
    """
    Save the binary artifacts of the pipeline to the specified local path.

    Args:
        path: The local path to save the pipeline
        pipeline: Transformers pipeline instance
        flavor_config: The flavor configuration constructed for the pipeline
        processor: Optional processor instance to save alongside the pipeline
    """
    model = get_peft_base_model(pipeline.model) if is_peft_model(pipeline.model) else pipeline.model
    model.save_pretrained(save_directory=path.joinpath(_MODEL_BINARY_FILE_NAME), max_shard_size=MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE.get())
    component_dir = path.joinpath(_COMPONENTS_BINARY_DIR_NAME)
    for name in flavor_conf.get(FlavorKey.COMPONENTS, []):
        getattr(pipeline, name).save_pretrained(component_dir.joinpath(name))
    if processor:
        processor.save_pretrained(component_dir.joinpath(_PROCESSOR_BINARY_DIR_NAME))