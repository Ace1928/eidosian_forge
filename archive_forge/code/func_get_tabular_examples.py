from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def get_tabular_examples(model_name: str) -> dict[str, list[float]]:
    readme = httpx.get(f'https://huggingface.co/{model_name}/resolve/main/README.md')
    if readme.status_code != 200:
        warnings.warn(f'Cannot load examples from README for {model_name}', UserWarning)
        example_data = {}
    else:
        yaml_regex = re.search('(?:^|[\r\n])---[\n\r]+([\\S\\s]*?)[\n\r]+---([\n\r]|$)', readme.text)
        if yaml_regex is None:
            example_data = {}
        else:
            example_yaml = next(yaml.safe_load_all(readme.text[:yaml_regex.span()[-1]]))
            example_data = example_yaml.get('widget', {}).get('structuredData', {})
    if not example_data:
        raise ValueError(f'No example data found in README.md of {model_name} - Cannot build gradio demo. See the README.md here: https://huggingface.co/scikit-learn/tabular-playground/blob/main/README.md for a reference on how to provide example data to your model.')
    for data in example_data.values():
        for i, val in enumerate(data):
            if isinstance(val, float) and math.isnan(val):
                data[i] = 'NaN'
    return example_data