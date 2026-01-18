from typing import Any, Dict, Optional
from PIL import Image
from gradio import components
def is_transformers_pipeline_type(pipeline, class_name: str):
    cls = getattr(transformers, class_name, None)
    return cls and isinstance(pipeline, cls)