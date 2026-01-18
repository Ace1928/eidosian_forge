from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
import wandb
def get_mean_confidence_map(classes: List, confidence: List, class_id_to_label: Dict) -> Dict[str, float]:
    """Get Mean-confidence map from the predictions to be logged into a `wandb.Table`."""
    confidence_map = {v: [] for _, v in class_id_to_label.items()}
    for class_idx, confidence_value in zip(classes, confidence):
        confidence_map[class_id_to_label[class_idx]].append(confidence_value)
    updated_confidence_map = {}
    for label, confidence_list in confidence_map.items():
        if len(confidence_list) > 0:
            updated_confidence_map[label] = sum(confidence_list) / len(confidence_list)
        else:
            updated_confidence_map[label] = 0
    return updated_confidence_map