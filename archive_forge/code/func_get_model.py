from typing import Type
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config import ModelConfig, DeviceConfig
from vllm.model_executor.models import ModelRegistry
def get_model(model_config: ModelConfig, device_config: DeviceConfig, **kwargs) -> nn.Module:
    from transformers_neuronx.config import NeuronConfig, ContinuousBatchingConfig
    parallel_config = kwargs.get('parallel_config')
    scheduler_config = kwargs.get('scheduler_config')
    model_class = _get_model_architecture(model_config.hf_config)
    linear_method = None
    model = model_class(model_config.hf_config, linear_method)
    continuous_batching_config = ContinuousBatchingConfig(batch_size_for_shared_caches=scheduler_config.max_num_seqs)
    neuron_config = NeuronConfig(continuous_batching=continuous_batching_config)
    model.load_weights(model_config.model, model_config.download_dir, model_config.load_format, model_config.revision, tp_degree=parallel_config.neuron_tp_degree, amp=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype], neuron_config=neuron_config, context_length_estimate=[scheduler_config.max_model_len], n_positions=[scheduler_config.max_model_len], batch_size=scheduler_config.max_num_seqs)
    return model.eval()