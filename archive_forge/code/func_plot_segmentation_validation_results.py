from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils.ops import scale_image
import wandb
from wandb.integration.ultralytics.bbox_utils import (
def plot_segmentation_validation_results(dataloader, class_label_map, model_name: str, predictor: SegmentationPredictor, table: wandb.Table, max_validation_batches: int, epoch: Optional[int]=None):
    data_idx = 0
    num_dataloader_batches = len(dataloader.dataset) // dataloader.batch_size
    max_validation_batches = min(max_validation_batches, num_dataloader_batches)
    for batch_idx, batch in enumerate(dataloader):
        prediction_results = predictor(batch['im_file'])
        progress_bar_result_iterable = tqdm(enumerate(prediction_results), total=len(prediction_results), desc=f'Generating Visualizations for batch-{batch_idx + 1}/{max_validation_batches}')
        for img_idx, prediction_result in progress_bar_result_iterable:
            prediction_result = prediction_result.to('cpu')
            _, prediction_mask_data, prediction_box_data, mean_confidence_map = plot_mask_predictions(prediction_result, model_name)
            try:
                ground_truth_data = get_ground_truth_bbox_annotations(img_idx, batch['im_file'][img_idx], batch, class_label_map)
                wandb_image = wandb.Image(batch['im_file'][img_idx], boxes={'ground-truth': {'box_data': ground_truth_data, 'class_labels': class_label_map}, 'predictions': prediction_box_data}, masks=prediction_mask_data)
                table_rows = [data_idx, batch_idx, wandb_image, mean_confidence_map, prediction_result.speed]
                table_rows = [epoch] + table_rows if epoch is not None else table_rows
                table_rows = [model_name] + table_rows
                table.add_data(*table_rows)
                data_idx += 1
            except TypeError:
                pass
        if batch_idx + 1 == max_validation_batches:
            break
    return table