import torch
from torchvision.models import resnet34
from transformers import (
from accelerate import PartialState
from accelerate.inference import prepare_pippy
from accelerate.utils import DistributedType, send_to_device, set_seed
def get_model_and_data_for_text(model_name, device, num_processes: int=2):
    initializer, config, seq_len = model_to_config[model_name]
    config_args = {}
    model_config = config(**config_args)
    model = initializer(model_config)
    return (model, torch.randint(low=0, high=model_config.vocab_size, size=(num_processes, seq_len), device=device, dtype=torch.int64, requires_grad=False))