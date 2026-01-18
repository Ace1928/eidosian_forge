import inspect
from dataclasses import asdict
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from peft import (
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq
from llama_recipes.configs import datasets, lora_config, llama_adapter_config, prefix_config, train_config
from llama_recipes.data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler
from llama_recipes.utils.dataset_utils import DATASET_PREPROC
def get_dataloader_kwargs(train_config, dataset, tokenizer, mode):
    kwargs = {}
    batch_size = train_config.batch_size_training if mode == 'train' else train_config.val_batch_size
    if train_config.batching_strategy == 'padding':
        if train_config.enable_fsdp:
            kwargs['batch_sampler'] = DistributedLengthBasedBatchSampler(dataset, batch_size=batch_size, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=mode == 'train')
        else:
            kwargs['batch_sampler'] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode == 'train')
        kwargs['collate_fn'] = DataCollatorForSeq2Seq(tokenizer)
    elif train_config.batching_strategy == 'packing':
        if train_config.enable_fsdp:
            kwargs['sampler'] = DistributedSampler(dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=mode == 'train', drop_last=True)
        kwargs['batch_size'] = batch_size
        kwargs['drop_last'] = True
        kwargs['collate_fn'] = default_data_collator
    else:
        raise ValueError(f'Unknown batching strategy: {train_config.batching_strategy}')
    return kwargs