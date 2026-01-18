from __future__ import annotations
import importlib.util
import logging
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Extra
@classmethod
def from_model_id(cls, model_id: str, task: str, backend: str='default', device: Optional[int]=-1, device_map: Optional[str]=None, model_kwargs: Optional[dict]=None, pipeline_kwargs: Optional[dict]=None, batch_size: int=DEFAULT_BATCH_SIZE, **kwargs: Any) -> HuggingFacePipeline:
    """Construct the pipeline object from model_id and task."""
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise ValueError('Could not import transformers python package. Please install it with `pip install transformers`.')
    _model_kwargs = model_kwargs or {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
    try:
        if task == 'text-generation':
            if backend == 'openvino':
                try:
                    from optimum.intel.openvino import OVModelForCausalLM
                except ImportError:
                    raise ValueError("Could not import optimum-intel python package. Please install it with: pip install 'optimum[openvino,nncf]' ")
                try:
                    model = OVModelForCausalLM.from_pretrained(model_id, **_model_kwargs)
                except Exception:
                    model = OVModelForCausalLM.from_pretrained(model_id, export=True, **_model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, **_model_kwargs)
        elif task in ('text2text-generation', 'summarization', 'translation'):
            if backend == 'openvino':
                try:
                    from optimum.intel.openvino import OVModelForSeq2SeqLM
                except ImportError:
                    raise ValueError("Could not import optimum-intel python package. Please install it with: pip install 'optimum[openvino,nncf]' ")
                try:
                    model = OVModelForSeq2SeqLM.from_pretrained(model_id, **_model_kwargs)
                except Exception:
                    model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, **_model_kwargs)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **_model_kwargs)
        else:
            raise ValueError(f'Got invalid task {task}, currently only {VALID_TASKS} are supported')
    except ImportError as e:
        raise ValueError(f'Could not load the {task} model due to missing dependencies.') from e
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = model.config.eos_token_id
    if (getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False)) and device is not None and (backend == 'default'):
        logger.warning(f'Setting the `device` argument to None from {device} to avoid the error caused by attempting to move the model that was already loaded on the GPU using the Accelerate module to the same or another device.')
        device = None
    if device is not None and importlib.util.find_spec('torch') is not None and (backend == 'default'):
        import torch
        cuda_device_count = torch.cuda.device_count()
        if device < -1 or device >= cuda_device_count:
            raise ValueError(f'Got device=={device}, device is required to be within [-1, {cuda_device_count})')
        if device_map is not None and device < 0:
            device = None
        if device is not None and device < 0 and (cuda_device_count > 0):
            logger.warning('Device has %d GPUs available. Provide device={deviceId} to `from_model_id` to use availableGPUs for execution. deviceId is -1 (default) for CPU and can be a positive integer associated with CUDA device id.', cuda_device_count)
    if device is not None and device_map is not None and (backend == 'openvino'):
        logger.warning("Please set device for OpenVINO through: 'model_kwargs'")
    if 'trust_remote_code' in _model_kwargs:
        _model_kwargs = {k: v for k, v in _model_kwargs.items() if k != 'trust_remote_code'}
    _pipeline_kwargs = pipeline_kwargs or {}
    pipeline = hf_pipeline(task=task, model=model, tokenizer=tokenizer, device=device, device_map=device_map, batch_size=batch_size, model_kwargs=_model_kwargs, **_pipeline_kwargs)
    if pipeline.task not in VALID_TASKS:
        raise ValueError(f'Got invalid task {pipeline.task}, currently only {VALID_TASKS} are supported')
    return cls(pipeline=pipeline, model_id=model_id, model_kwargs=_model_kwargs, pipeline_kwargs=_pipeline_kwargs, batch_size=batch_size, **kwargs)