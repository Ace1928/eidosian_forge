import sys
import warnings
import torch
from torch.onnx import symbolic_opset11 as opset11
from torch.onnx.symbolic_helper import parse_args
@parse_args('v', 'v', 'f')
def symbolic_multi_label_nms(g, boxes, scores, iou_threshold):
    boxes = opset11.unsqueeze(g, boxes, 0)
    scores = opset11.unsqueeze(g, opset11.unsqueeze(g, scores, 0), 0)
    max_output_per_class = g.op('Constant', value_t=torch.tensor([sys.maxsize], dtype=torch.long))
    iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
    nms_out = g.op('NonMaxSuppression', g.op('Cast', boxes, to_i=torch.onnx.TensorProtoDataType.FLOAT), g.op('Cast', scores, to_i=torch.onnx.TensorProtoDataType.FLOAT), max_output_per_class, iou_threshold)
    return opset11.squeeze(g, opset11.select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))), 1)