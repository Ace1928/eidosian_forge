import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def post_process_panoptic(self, outputs, processed_sizes, target_sizes=None, is_thing_map=None, threshold=0.85):
    """
        Converts the output of [`DetrForSegmentation`] into actual panoptic predictions. Only supports PyTorch.

        Args:
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            processed_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`):
                Torch Tensor (or list) containing the size (h, w) of each image of the batch, i.e. the size after data
                augmentation but before batching.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`, *optional*):
                Torch Tensor (or list) corresponding to the requested final size `(height, width)` of each prediction.
                If left to None, it will default to the `processed_sizes`.
            is_thing_map (`torch.Tensor` of shape `(batch_size, 2)`, *optional*):
                Dictionary mapping class indices to either True or False, depending on whether or not they are a thing.
                If not set, defaults to the `is_thing_map` of COCO panoptic.
            threshold (`float`, *optional*, defaults to 0.85):
                Threshold to use to filter out queries.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing a PNG string and segments_info values for
            an image in the batch as predicted by the model.
        """
    logger.warning_once('`post_process_panoptic is deprecated and will be removed in v5 of Transformers, please use `post_process_panoptic_segmentation`.')
    if target_sizes is None:
        target_sizes = processed_sizes
    if len(processed_sizes) != len(target_sizes):
        raise ValueError('Make sure to pass in as many processed_sizes as target_sizes')
    if is_thing_map is None:
        is_thing_map = {i: i <= 90 for i in range(201)}
    out_logits, raw_masks, raw_boxes = (outputs.logits, outputs.pred_masks, outputs.pred_boxes)
    if not len(out_logits) == len(raw_masks) == len(target_sizes):
        raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits and masks')
    empty_label = out_logits.shape[-1] - 1
    preds = []

    def to_tuple(tup):
        if isinstance(tup, tuple):
            return tup
        return tuple(tup.cpu().tolist())
    for cur_logits, cur_masks, cur_boxes, size, target_size in zip(out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes):
        cur_scores, cur_labels = cur_logits.softmax(-1).max(-1)
        keep = cur_labels.ne(empty_label) & (cur_scores > threshold)
        cur_scores = cur_scores[keep]
        cur_labels = cur_labels[keep]
        cur_masks = cur_masks[keep]
        cur_masks = nn.functional.interpolate(cur_masks[:, None], to_tuple(size), mode='bilinear').squeeze(1)
        cur_boxes = center_to_corners_format(cur_boxes[keep])
        h, w = cur_masks.shape[-2:]
        if len(cur_boxes) != len(cur_labels):
            raise ValueError('Not as many boxes as there are classes')
        cur_masks = cur_masks.flatten(1)
        stuff_equiv_classes = defaultdict(lambda: [])
        for k, label in enumerate(cur_labels):
            if not is_thing_map[label.item()]:
                stuff_equiv_classes[label.item()].append(k)

        def get_ids_area(masks, scores, dedup=False):
            m_id = masks.transpose(0, 1).softmax(-1)
            if m_id.shape[-1] == 0:
                m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
            else:
                m_id = m_id.argmax(-1).view(h, w)
            if dedup:
                for equiv in stuff_equiv_classes.values():
                    if len(equiv) > 1:
                        for eq_id in equiv:
                            m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
            final_h, final_w = to_tuple(target_size)
            seg_img = PIL.Image.fromarray(id_to_rgb(m_id.view(h, w).cpu().numpy()))
            seg_img = seg_img.resize(size=(final_w, final_h), resample=PILImageResampling.NEAREST)
            np_seg_img = torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes()))
            np_seg_img = np_seg_img.view(final_h, final_w, 3)
            np_seg_img = np_seg_img.numpy()
            m_id = torch.from_numpy(rgb_to_id(np_seg_img))
            area = []
            for i in range(len(scores)):
                area.append(m_id.eq(i).sum().item())
            return (area, seg_img)
        area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
        if cur_labels.numel() > 0:
            while True:
                filtered_small = torch.as_tensor([area[i] <= 4 for i, c in enumerate(cur_labels)], dtype=torch.bool, device=keep.device)
                if filtered_small.any().item():
                    cur_scores = cur_scores[~filtered_small]
                    cur_labels = cur_labels[~filtered_small]
                    cur_masks = cur_masks[~filtered_small]
                    area, seg_img = get_ids_area(cur_masks, cur_scores)
                else:
                    break
        else:
            cur_labels = torch.ones(1, dtype=torch.long, device=cur_labels.device)
        segments_info = []
        for i, a in enumerate(area):
            cat = cur_labels[i].item()
            segments_info.append({'id': i, 'isthing': is_thing_map[cat], 'category_id': cat, 'area': a})
        del cur_labels
        with io.BytesIO() as out:
            seg_img.save(out, format='PNG')
            predictions = {'png_string': out.getvalue(), 'segments_info': segments_info}
        preds.append(predictions)
    return preds