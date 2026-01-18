import importlib
import logging
from typing import Any, Callable, List, Optional
from langchain_community.embeddings.self_hosted import SelfHostedEmbeddings
def load_embedding_model(model_id: str, instruct: bool=False, device: int=0) -> Any:
    """Load the embedding model."""
    if not instruct:
        import sentence_transformers
        client = sentence_transformers.SentenceTransformer(model_id)
    else:
        from InstructorEmbedding import INSTRUCTOR
        client = INSTRUCTOR(model_id)
    if importlib.util.find_spec('torch') is not None:
        import torch
        cuda_device_count = torch.cuda.device_count()
        if device < -1 or device >= cuda_device_count:
            raise ValueError(f'Got device=={device}, device is required to be within [-1, {cuda_device_count})')
        if device < 0 and cuda_device_count > 0:
            logger.warning('Device has %d GPUs available. Provide device={deviceId} to `from_model_id` to use availableGPUs for execution. deviceId is -1 for CPU and can be a positive integer associated with CUDA device id.', cuda_device_count)
        client = client.to(device)
    return client