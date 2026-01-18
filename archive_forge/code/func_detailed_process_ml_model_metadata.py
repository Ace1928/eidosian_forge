import json
import hashlib
import logging
from typing import List, Any, Dict
def detailed_process_ml_model_metadata(ml_model_metadata: Dict) -> str:
    model_name = ml_model_metadata.get('ModelName', 'No Model Name')
    model_version = ml_model_metadata.get('ModelVersion', 'No Version Info')
    model_description = ml_model_metadata.get('ModelDescription', 'No Description')
    return f'Model Name: {model_name}, Version: {model_version}, Description: {model_description}'