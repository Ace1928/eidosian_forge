import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import onnx
from onnx import ModelProto
from ..utils import logging
from .transformations_utils import (
def check_and_save_model(model: onnx.ModelProto, save_path: Optional[Union[str, Path]]):
    if model.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF:
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            if 'No Op registered for' in str(e):
                pass
            else:
                raise e
        if save_path:
            save_path = Path(save_path).as_posix()
            external_file_name = os.path.basename(save_path) + '_data'
            external_path = os.path.join(os.path.dirname(save_path), external_file_name)
            if save_path.endswith('.onnx') and os.path.isfile(save_path):
                os.remove(save_path)
            if os.path.isfile(external_path):
                os.remove(external_path)
            onnx.save(model, save_path)
    elif save_path is not None:
        save_path = Path(save_path).as_posix()
        external_file_name = os.path.basename(save_path) + '_data'
        external_path = os.path.join(os.path.dirname(save_path), external_file_name)
        if save_path.endswith('.onnx') and os.path.isfile(save_path):
            os.remove(save_path)
        if os.path.isfile(external_path):
            os.remove(external_path)
        onnx.save(model, save_path, save_as_external_data=True, all_tensors_to_one_file=True, location=external_file_name)
        try:
            onnx.checker.check_model(save_path)
        except Exception as e:
            if 'No Op registered for' in str(e):
                pass
            else:
                raise e
    else:
        logger.info('Merged ONNX model exceeds 2GB, the model will not be checked without `save_path` given.')