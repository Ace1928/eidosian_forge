from typing import Any, Optional
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.pose import PosePredictor
from ultralytics.utils.plotting import Annotator
import wandb
from wandb.integration.ultralytics.bbox_utils import (
def plot_pose_validation_results(dataloader, class_label_map, model_name: str, predictor: PosePredictor, visualize_skeleton: bool, table: wandb.Table, max_validation_batches: int, epoch: Optional[int]=None) -> wandb.Table:
    data_idx = 0
    num_dataloader_batches = len(dataloader.dataset) // dataloader.batch_size
    max_validation_batches = min(max_validation_batches, num_dataloader_batches)
    for batch_idx, batch in enumerate(dataloader):
        prediction_results = predictor(batch['im_file'])
        progress_bar_result_iterable = tqdm(enumerate(prediction_results), total=len(prediction_results), desc=f'Generating Visualizations for batch-{batch_idx + 1}/{max_validation_batches}')
        for img_idx, prediction_result in progress_bar_result_iterable:
            prediction_result = prediction_result.to('cpu')
            table_row = plot_pose_predictions(prediction_result, model_name, visualize_skeleton)
            ground_truth_image = wandb.Image(annotate_keypoint_batch(batch['im_file'][img_idx], batch['keypoints'][img_idx], visualize_skeleton), boxes={'ground-truth': {'box_data': get_ground_truth_bbox_annotations(img_idx, batch['im_file'][img_idx], batch, class_label_map), 'class_labels': class_label_map}})
            table_row = [data_idx, batch_idx, ground_truth_image] + table_row[1:]
            table_row = [epoch] + table_row if epoch is not None else table_row
            table_row = [model_name] + table_row
            table.add_data(*table_row)
            data_idx += 1
        if batch_idx + 1 == max_validation_batches:
            break
    return table